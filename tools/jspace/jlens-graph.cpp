#include "jlens.h"

#include <cmath>
#include <cstdio>

// GLM-4 dense forward graph with per-block residual capture.
//
// Every op used here has a verified ggml backward path (CPU + CUDA):
//   get_rows / rms_norm / mul / add / mul_mat / rope_ext (NORM mode) /
//   repeat (GQA) / transpose+cont / soft_max_ext (max_bias=0) / swiglu_split /
//   reshape / view
// No flash attention, no kv cache, no fused gate_up GLU (split form required
// for the GLU backward case in ggml_compute_backward).

static ggml_tensor * rms_norm_mul(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// [D, Hkv, n] -> [D, H, n] with the mul_mat GQA head map (q head h uses kv head h / (H/Hkv))
static ggml_tensor * gqa_expand(ggml_context * ctx, ggml_tensor * x, int64_t n_head) {
    const int64_t D = x->ne[0], Hkv = x->ne[1], n = x->ne[2];
    ggml_tensor * x4 = ggml_reshape_4d(ctx, x, D, 1, Hkv, n);
    x4 = ggml_repeat_4d(ctx, x4, D, n_head / Hkv, Hkv, n);
    return ggml_reshape_3d(ctx, x4, D, n_head, n);
}

// one GLM-4 dense block: post-norm attention + swiglu MLP with residuals.
// dt_src mode: instead of the raw detach sources, emits the FINAL detached leaf
// values computed in-graph (rms_inv = 1/sqrt(mean(x^2)+eps), sigmoid(gate),
// silu(gate), up) so the LRP graph's leaf inputs can alias them directly —
// no separate leaf buffers, no host roundtrip. dt_one/dt_eps are [1] f32
// constant leaves (1.0, norm_eps) supplied by the caller.
// ckv: chunk mode (see jlens_upper_chunk); the block's queries are the n_tokens chunk positions and
// keys/values span ckv->k_kv. kv_out: receives the block's keys (after rope) and values, [D, Hkv, n_tokens].
static ggml_tensor * glm_block(ggml_context * ctx, const jlens_weights & m, int il, int n_tokens,
                               ggml_tensor * inpL, ggml_tensor * pos, ggml_tensor * mask,
                               bool use_flash_attn,
                               jlens_lrp_leaves * lrp = nullptr,
                               std::vector<ggml_tensor *> * dt_src = nullptr,
                               ggml_tensor * dt_one = nullptr, ggml_tensor * dt_eps = nullptr,
                               jlens_chunk_kv * ckv = nullptr, ggml_tensor ** kv_out = nullptr) {
    const auto & hp = m.hparams;
    const int64_t n_embd      = hp.n_embd;
    const int64_t n_head      = hp.n_head;
    const int64_t n_head_kv   = hp.n_head_kv;
    const int64_t n_embd_head = hp.n_embd_head;
    const int64_t n_ff        = hp.n_ff;
    const float   kq_scale    = 1.0f / sqrtf((float) n_embd_head);

    ggml_tensor * inpSA = inpL;

    // rms-norm site k. LRP mode applies the LN-rule (detached denominator) by
    // construction; dt_src mode emits the detached rms_inv for harvesting.
    int n_norm_site = 0;
    auto norm_site = [&](ggml_tensor * x, ggml_tensor * w) {
        if (dt_src) {
            ggml_tensor * ri = ggml_div(ctx, dt_one,
                ggml_sqrt(ctx, ggml_add(ctx, ggml_mean(ctx, ggml_sqr(ctx, x)), dt_eps)));
            char name[64];
            snprintf(name, sizeof(name), "jlens_dt_%d_rmsinv_%d", il, n_norm_site);
            ggml_set_name(ri, name);
            ggml_set_output(ri);
            dt_src->push_back(ri);
        }
        ggml_tensor * y;
        if (lrp) {
            y = ggml_mul(ctx, ggml_mul(ctx, x, lrp->rms_inv[n_norm_site]), w);
        } else {
            y = ggml_mul(ctx, ggml_rms_norm(ctx, x, hp.norm_eps), w);
        }
        n_norm_site++;
        return y;
    };

    // pre-attention norm
    ggml_tensor * cur = norm_site(inpL, m.blk("blk.%d.attn_norm.weight", il));

    // qkv
    ggml_tensor * q = ggml_mul_mat(ctx, m.blk("blk.%d.attn_q.weight", il), cur); // [n_embd, n]
    ggml_tensor * k = ggml_mul_mat(ctx, m.blk("blk.%d.attn_k.weight", il), cur); // [n_embd_head*n_head_kv, n]
    ggml_tensor * v = ggml_mul_mat(ctx, m.blk("blk.%d.attn_v.weight", il), cur);

    q = ggml_reshape_3d(ctx, q, n_embd_head, n_head,    n_tokens);
    k = ggml_reshape_3d(ctx, k, n_embd_head, n_head_kv, n_tokens);
    v = ggml_reshape_3d(ctx, v, n_embd_head, n_head_kv, n_tokens);

    // rope (NORM mode 0, partial rotation n_rot)
    q = ggml_rope_ext(ctx, q, pos, nullptr, hp.n_rot, /*mode=*/0, hp.n_ctx_train,
                      hp.rope_freq_base, /*freq_scale=*/1.0f, /*ext_factor=*/0.0f,
                      /*attn_factor=*/1.0f, /*beta_fast=*/32.0f, /*beta_slow=*/1.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, hp.n_rot, 0, hp.n_ctx_train,
                      hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    if (kv_out) {
        kv_out[0] = k;
        kv_out[1] = v;
    }

    // chunk mode: swap in the keys/values of all positions [0, n_kv)
    ggml_tensor * k_live = k, * v_live = v;
    if (ckv) {
        GGML_ASSERT(!use_flash_attn);
        ckv->k_live = k;
        ckv->v_live = v;
        if (ckv->now) {
            k = ckv->k_kv;
            v = ckv->v_kv;
        } else {
            // x 1: the grad that reaches the SET comes back through a permute (strided); the MUL backward
            // makes it contiguous, which the SET backward (ggml_acc) requires
            k = ggml_mul(ctx, ggml_set_1d(ctx, ckv->k_kv, k, ckv->a * ckv->k_kv->nb[2]), ckv->one);
            v = ggml_mul(ctx, ggml_set_1d(ctx, ckv->v_kv, v, ckv->a * ckv->v_kv->nb[2]), ckv->one);
        }
    }

    // GQA: ggml_mul_mat broadcasts a's head dims (b head i maps to a head i / (Hb/Ha),
    // contiguous groups) and the MUL_MAT backward handles the broadcast, so no manual
    // kv-head expansion is needed.
    ggml_tensor * kqv;
    if (use_flash_attn) {
        // forward-only path: flash attention keeps the score matrix off VRAM.
        // layout [n_embd_head, n_tokens, n_head]; k/v + mask cast to f16, q stays
        // f32 (CUDA mma kernels assert Q->type == F32; matches llama's own usage).
        // GQA (n_head_kv < n_head) handled by the kernel. No backward support.
        ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        kp = ggml_cast(ctx, kp, GGML_TYPE_F16);
        vp = ggml_cast(ctx, vp, GGML_TYPE_F16);
        ggml_tensor * fmask = ggml_cast(ctx, mask, GGML_TYPE_F16); // flash attn requires f16 mask
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, qp, kp, vp, fmask, kq_scale,
                                               /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        // flash_attn_ext output is ALREADY permuted back to
        // [n_embd_head, n_head, n_q] (ggml.c: ne = {v->ne[0], q->ne[2], q->ne[1], ...})
        // — do NOT permute again
        kqv = ggml_reshape_2d(ctx, fa, n_embd, n_tokens);
    } else {
        // to [n_embd_head, n_tokens, n_head] layout (heads last) for batched attention matmuls
        ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

        // attention scores [n_kv, n_q, n_head] (k broadcast 2 -> 48 heads)
        ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);
        ggml_prec_set_acc(kq, GGML_PREC_F32); // matches llama-graph.cpp's GLM4 handling

        // now mode: each query sees the frozen keys/values, except its own key/value, which is live.
        // The diagonal terms add q.(k_live - k_frozen) and p_ii (v_live - v_frozen): value ~0, but
        // the gradient reaches k_live/v_live through the query's own position only.
        const bool diag = ckv && ckv->now;
        const size_t a_off = diag ? (size_t) ckv->a : 0;
        if (diag) {
            ggml_tensor * k_fz = ggml_view_3d(ctx, ckv->k_kv, n_embd_head, n_head_kv, n_tokens,
                                              ckv->k_kv->nb[1], ckv->k_kv->nb[2], a_off * ckv->k_kv->nb[2]);
            ggml_tensor * dk = gqa_expand(ctx, ggml_sub(ctx, k_live, k_fz), n_head); // [D, H, n_q]
            ggml_tensor * s_self = ggml_sum_rows(ctx, ggml_mul(ctx, q, dk));          // [1, H, n_q]
            s_self = ggml_cont(ctx, ggml_permute(ctx, s_self, 0, 2, 1, 3));          // [1, n_q, H]
            // diagonal of kq: element (a + i, i, h), so row stride nb[1] + nb[0]
            kq = ggml_acc(ctx, kq, s_self, kq->nb[1] + kq->nb[0], kq->nb[2], kq->nb[3], a_off * kq->nb[0]);
        }
        kq = ggml_soft_max_ext(ctx, kq, mask, kq_scale, /*max_bias=*/0.0f);

        // weighted values: [n_embd_head, n_q, n_head]
        // nb: build vt with a single permute+cont from v. A cont->transpose->cont chain
        // (or cont+mul_mat on a transposed view) makes the backward pass hand a
        // non-contiguous grad view to a CONT node, which ggml asserts against.
        ggml_tensor * vt  = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [n_kv, n_embd_head, n_head_kv]
        kqv = ggml_mul_mat(ctx, vt, kq);                        // [n_embd_head, n_q, n_head]
        if (diag) {
            ggml_tensor * v_fz = ggml_view_3d(ctx, ckv->v_kv, n_embd_head, n_head_kv, n_tokens,
                                              ckv->v_kv->nb[1], ckv->v_kv->nb[2], a_off * ckv->v_kv->nb[2]);
            ggml_tensor * dv = gqa_expand(ctx, ggml_sub(ctx, v_live, v_fz), n_head);   // [D, H, n_q]
            dv = ggml_cont(ctx, ggml_permute(ctx, dv, 0, 2, 1, 3));                   // [D, n_q, H]
            ggml_tensor * p_ii = ggml_cont(ctx, ggml_view_3d(ctx, kq, 1, n_tokens, n_head,
                                                             kq->nb[1] + kq->nb[0], kq->nb[2], a_off * kq->nb[0]));
            kqv = ggml_add(ctx, kqv, ggml_mul(ctx, dv, p_ii));
        }

        // merge heads -> [n_embd, n_q]
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));       // [n_embd_head, n_head, n_q]
        kqv = ggml_reshape_2d(ctx, kqv, n_embd, n_tokens);
    }

    cur = ggml_mul_mat(ctx, m.blk("blk.%d.attn_output.weight", il), kqv);
    ggml_prec_set_acc(cur, GGML_PREC_F32); // matches llama-graph.cpp's GLM4 handling

    // post-attention norm, residual
    cur = norm_site(cur, m.blk("blk.%d.post_attention_norm.weight", il));
    ggml_tensor * ffn_inp = ggml_add(ctx, cur, inpSA);

    // pre-MLP norm
    cur = norm_site(ffn_inp, m.blk("blk.%d.ffn_norm.weight", il));

    // combined gate|up projection [2*n_ff, n]; split for swiglu_split (backward needs split form)
    ggml_tensor * up_out = ggml_mul_mat(ctx, m.blk("blk.%d.ffn_up.weight", il), cur);
    ggml_tensor * gate   = ggml_cont(ctx, ggml_view_2d(ctx, up_out, n_ff, n_tokens,
                                    up_out->nb[1], 0));
    ggml_tensor * up     = ggml_cont(ctx, ggml_view_2d(ctx, up_out, n_ff, n_tokens,
                                    up_out->nb[1], n_ff * up_out->nb[0]));
    if (dt_src) {
        char name[64];
        snprintf(name, sizeof(name), "jlens_dt_%d_sig", il);
        ggml_tensor * sg = ggml_sigmoid(ctx, gate);
        ggml_set_name(sg, name); ggml_set_output(sg); dt_src->push_back(sg);
        snprintf(name, sizeof(name), "jlens_dt_%d_silu", il);
        ggml_tensor * sl = ggml_silu(ctx, gate);
        ggml_set_name(sl, name); ggml_set_output(sl); dt_src->push_back(sl);
        snprintf(name, sizeof(name), "jlens_dt_%d_up", il);
        ggml_set_name(up, name);   ggml_set_output(up);   dt_src->push_back(up);
    }
    if (lrp) {
        // identity + half rule by construction:
        //   y = 0.5*(g*sig_leaf)*up_leaf + 0.5*silu_leaf*u
        // autodiff gives dg = 0.5*dy*sig*u (identity rule on silu, half rule on the
        // product) and du = 0.5*dy*silu(g) (half rule). Forward value unchanged.
        ggml_tensor * a = ggml_mul(ctx, ggml_mul(ctx, gate, lrp->sig_gate), lrp->up);
        ggml_tensor * b = ggml_mul(ctx, lrp->silu_gate, up);
        cur = ggml_add(ctx, ggml_scale(ctx, a, 0.5f), ggml_scale(ctx, b, 0.5f));
    } else {
        cur = ggml_swiglu_split(ctx, gate, up);                      // silu(gate) * up, [n_ff, n]
    }
    cur = ggml_mul_mat(ctx, m.blk("blk.%d.ffn_down.weight", il), cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32); // matches llama-graph.cpp's GLM4 handling

    // post-MLP norm, residual
    cur = norm_site(cur, m.blk("blk.%d.post_ffw_norm.weight", il));
    return ggml_add(ctx, cur, ffn_inp);
}

bool jlens_build_block(const jlens_weights & m, int il, int n_tokens, jlens_block & out) {
    const auto & hp = m.hparams;

    const int    gsize    = 512;
    const int    n_tensors= 440;
    const size_t ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(gsize, true);
    struct ggml_init_params iparams = { ctx_size, nullptr, /*no_alloc=*/true };
    out.gctx = ggml_init(iparams);
    ggml_context * ctx = out.gctx;

    out.gf = ggml_new_graph_custom(ctx, gsize, /*grads=*/true);

    out.h_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, n_tokens);
    ggml_set_name(out.h_in, "jlens_blk_h_in");
    ggml_set_input(out.h_in);

    out.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(out.pos, "jlens_blk_pos");
    ggml_set_input(out.pos);

    out.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(out.mask, "jlens_blk_mask");
    ggml_set_input(out.mask);

    // same param-coverage hack as the full graph; grad capture point is x
    out.zero_pad = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(out.zero_pad, "jlens_blk_zero_pad");
    ggml_set_param(out.zero_pad);
    out.x = ggml_add(ctx, out.h_in, out.zero_pad);

    out.h_out = glm_block(ctx, m, il, n_tokens, out.x, out.pos, out.mask, /*use_flash_attn=*/false);
    ggml_set_name(out.h_out, "jlens_blk_h_out");
    out.h_out->flags |= GGML_TENSOR_FLAG_LOSS; // caller supplies the grad_accs seed
    ggml_build_forward_expand(out.gf, out.h_out);
    return true;
}


bool jlens_build_forward(const jlens_weights & m, int n_tokens, bool with_grad_flags, jlens_forward & out,
                         int perturb_layer, bool use_flash_attn, bool with_head,
                         const std::vector<ggml_tensor *> * lin_w) {
    if (with_grad_flags && use_flash_attn) {
        fprintf(stderr, "jlens_build_forward: flash attention has no backward path\n");
        return false;
    }
    const auto & hp = m.hparams;
    const int64_t n_embd      = hp.n_embd;
    const int64_t n_head      = hp.n_head;
    const int64_t n_head_kv   = hp.n_head_kv;
    const int64_t n_embd_head = hp.n_embd_head;
    const int64_t n_ff        = hp.n_ff;
    const float   kq_scale    = 1.0f / sqrtf((float) n_embd_head);

    // rough node count: forward ~40 ops/layer; with backward expansion the graph
    // grows to ~200 nodes/layer, so size everything generously when grads are on
    const int    gsize    = with_grad_flags ? hp.n_layer * 256 : hp.n_layer * 64;
    const int    n_tensors= with_grad_flags ? hp.n_layer * 220 : hp.n_layer * 64;
    const size_t ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(gsize, with_grad_flags);
    struct ggml_init_params iparams = { ctx_size, nullptr, /*no_alloc=*/true };
    out.gctx = ggml_init(iparams);
    ggml_context * ctx = out.gctx;

    out.gf = ggml_new_graph_custom(ctx, gsize, /*grads=*/with_grad_flags);

    out.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(out.tokens, "jlens_tokens");
    ggml_set_input(out.tokens);

    out.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(out.pos, "jlens_pos");
    ggml_set_input(out.pos);

    // causal mask, [n_kv, n_q]
    out.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(out.mask, "jlens_mask");
    ggml_set_input(out.mask);

    ggml_tensor * inpL = ggml_get_rows(ctx, m.get("token_embd.weight"), out.tokens); // [n_embd, n_tokens]

    if (with_grad_flags) {
        // ggml_build_backward_expand requires a PARAM leaf and marks grads_needed by
        // forward propagation from PARAMs. A scalar zero added to the embedding output
        // makes every node in the network a "descendant of a param" without computing
        // any real weight gradients (weights are quantized anyway, which ggml refuses
        // to differentiate). The caller must set it to 0.0f before computing.
        out.zero_pad = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_name(out.zero_pad, "jlens_zero_pad");
        ggml_set_param(out.zero_pad);
        inpL = ggml_add(ctx, inpL, out.zero_pad);
    }

    out.l_out.resize(hp.n_layer, nullptr);
    out.k_out.resize(hp.n_layer, nullptr);
    out.v_out.resize(hp.n_layer, nullptr);

    for (int il = 0; il < hp.n_layer; ++il) {
        ggml_tensor * kv[2] = {};
        ggml_tensor * cur = glm_block(ctx, m, il, n_tokens, inpL, out.pos, out.mask, use_flash_attn,
                                      nullptr, nullptr, nullptr, nullptr, nullptr, kv);
        out.k_out[il] = kv[0];
        out.v_out[il] = kv[1];

        {
            char name[64];
            snprintf(name, sizeof(name), "jlens_l_out_%d", il);
            ggml_set_name(cur, name);
            if (with_grad_flags && il == (int) hp.n_layer - 1) {
                // VJP seed point. ggml_set_loss() asserts scalar, but the flag is all
                // we need: the caller supplies a grad_accs entry of this shape and
                // never calls ggml_graph_reset.
                cur->flags |= GGML_TENSOR_FLAG_LOSS;
            }
            ggml_build_forward_expand(out.gf, cur);
        }

        out.l_out[il] = cur;
        if (lin_w && il < (int) lin_w->size() && (*lin_w)[il]) {
            ggml_tensor * mm = ggml_mul_mat(ctx, (*lin_w)[il], cur); // [n_lin, n_tokens]
            out.lin = out.lin ? ggml_add(ctx, out.lin, mm) : mm;
            ggml_build_forward_expand(out.gf, out.lin);
        }
        if (with_grad_flags && il == perturb_layer) {
            // additive probe input for finite-difference validation of the backward pass
            out.perturb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, n_tokens);
            ggml_set_name(out.perturb, "jlens_perturb");
            cur = ggml_add(ctx, cur, out.perturb);
        }
        inpL = cur;
    }

    if (with_head) {
        // full-model head: rms norm + output projection, [n_vocab, n_tokens]
        ggml_tensor * normed = rms_norm_mul(ctx, inpL, m.get("output_norm.weight"), hp.norm_eps);
        out.logits = ggml_mul_mat(ctx, m.get("output.weight"), normed);
        ggml_set_name(out.logits, "jlens_logits");
        ggml_build_forward_expand(out.gf, out.logits);
        ggml_set_output(out.logits);
    } else {
        out.logits = nullptr;
        ggml_build_forward_expand(out.gf, inpL);
    }
    return true;
}

bool jlens_build_forward_steer(const jlens_weights & m, int n_tokens, int steer_layer, jlens_steer_forward & out,
                               bool per_position) {
    const auto & hp = m.hparams;
    if (steer_layer < 0 || steer_layer >= hp.n_layer) {
        fprintf(stderr, "jlens_build_forward_steer: steer_layer out of range\n");
        return false;
    }

    const int    gsize    = hp.n_layer * 256;
    const int    n_tensors= hp.n_layer * 220;
    const size_t ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(gsize, true);
    struct ggml_init_params iparams = { ctx_size, nullptr, /*no_alloc=*/true };
    out.gctx = ggml_init(iparams);
    ggml_context * ctx = out.gctx;

    out.gf = ggml_new_graph_custom(ctx, gsize, /*grads=*/true);

    out.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(out.tokens, "jlens_tokens");
    ggml_set_input(out.tokens);

    out.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(out.pos, "jlens_pos");
    ggml_set_input(out.pos);

    out.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(out.mask, "jlens_mask");
    ggml_set_input(out.mask);

    out.steer = per_position ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, n_tokens)
                             : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
    ggml_set_name(out.steer, "jlens_steer");
    ggml_set_param(out.steer);

    ggml_tensor * inpL = ggml_get_rows(ctx, m.get("token_embd.weight"), out.tokens);
    out.l_out.resize(hp.n_layer, nullptr);
    for (int il = 0; il < hp.n_layer; ++il) {
        ggml_tensor * cur = glm_block(ctx, m, il, n_tokens, inpL, out.pos, out.mask, /*use_flash_attn=*/false);
        char name[64];
        snprintf(name, sizeof(name), "jlens_l_out_%d", il);
        ggml_set_name(cur, name);
        out.l_out[il] = cur;
        if (il == steer_layer) {
            cur = ggml_add(ctx, cur, out.steer); // broadcast over positions (or one slot per position)
        }
        inpL = cur;
    }

    ggml_tensor * normed = rms_norm_mul(ctx, inpL, m.get("output_norm.weight"), hp.norm_eps);
    out.logits = ggml_mul_mat(ctx, m.get("output.weight"), normed);
    ggml_set_name(out.logits, "jlens_logits");
    ggml_set_output(out.logits);

    out.labels = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out.logits->ne[0], n_tokens);
    ggml_set_name(out.labels, "jlens_labels");
    ggml_set_input(out.labels);

    out.loss = ggml_cross_entropy_loss(ctx, out.logits, out.labels);
    ggml_set_name(out.loss, "jlens_loss");
    ggml_set_output(out.loss);
    out.loss->flags |= GGML_TENSOR_FLAG_LOSS;
    ggml_build_forward_expand(out.gf, out.loss);
    return true;
}

ggml_cgraph * jlens_build_head(const jlens_weights & m, ggml_context * gctx, ggml_tensor ** x_in) {
    ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, m.hparams.n_embd, 1);
    ggml_set_name(x, "jlens_head_x");
    ggml_set_input(x);
    *x_in = x;

    ggml_tensor * normed = rms_norm_mul(gctx, x, m.get("output_norm.weight"), m.hparams.norm_eps);
    ggml_tensor * logits = ggml_mul_mat(gctx, m.get("output.weight"), normed);
    ggml_set_name(logits, "jlens_head_logits");
    ggml_set_output(logits);

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 8, false);
    ggml_build_forward_expand(gf, logits);
    return gf;
}

// phase 1: ordinary forward with in-graph detach transforms named + marked as
// outputs: jlens_dt_<il>_rmsinv_<k> [1, n_tokens], jlens_dt_<il>_sig /
// jlens_dt_<il>_silu / jlens_dt_<il>_up [n_ff, n_tokens].
// dt_one/dt_eps: [1] f32 constant leaves (1.0, norm_eps) owned by the caller.
bool jlens_build_forward_dt(const jlens_weights & m, int n_tokens, jlens_forward & out,
                            ggml_tensor * dt_one, ggml_tensor * dt_eps) {
    const auto & hp = m.hparams;

    const int    gsize    = hp.n_layer * 96;   // in-graph transforms add ~22 nodes/layer
    const int    n_tensors= hp.n_layer * 96;
    const size_t ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(gsize, false);
    struct ggml_init_params iparams = { ctx_size, nullptr, /*no_alloc=*/true };
    out.gctx = ggml_init(iparams);
    ggml_context * ctx = out.gctx;

    out.gf = ggml_new_graph_custom(ctx, gsize, /*grads=*/false);

    out.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(out.tokens, "jlens_tokens");
    ggml_set_input(out.tokens);

    out.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(out.pos, "jlens_pos");
    ggml_set_input(out.pos);

    out.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(out.mask, "jlens_mask");
    ggml_set_input(out.mask);

    ggml_tensor * inpL = ggml_get_rows(ctx, m.get("token_embd.weight"), out.tokens);

    std::vector<ggml_tensor *> dt_src;
    dt_src.reserve(hp.n_layer * 6);
    out.l_out.resize(hp.n_layer, nullptr);

    for (int il = 0; il < hp.n_layer; ++il) {
        const size_t dt_before = dt_src.size();
        ggml_tensor * cur = glm_block(ctx, m, il, n_tokens, inpL, out.pos, out.mask,
                                      /*use_flash_attn=*/false, /*lrp=*/nullptr, &dt_src,
                                      dt_one, dt_eps);
        // dt outputs are side branches (not ancestors of the forward), so they
        // must be explicitly expanded into the graph
        for (size_t i = dt_before; i < dt_src.size(); ++i) {
            ggml_build_forward_expand(out.gf, dt_src[i]);
        }
        char name[64];
        snprintf(name, sizeof(name), "jlens_l_out_%d", il);
        ggml_set_name(cur, name);
        ggml_build_forward_expand(out.gf, cur);
        if (il == 0) ggml_set_output(cur); // pinned for the fitter's dt self-check
        out.l_out[il] = cur;
        inpL = cur;
    }

    out.logits = ggml_mul_mat(ctx, m.get("output.weight"),
                              rms_norm_mul(ctx, inpL, m.get("output_norm.weight"), hp.norm_eps));
    ggml_set_name(out.logits, "jlens_logits");
    ggml_build_forward_expand(out.gf, out.logits);
    ggml_set_output(out.logits);
    return true;
}

// phase 2: LRP-structured graph (LN-rule + identity/half rule via detached leaves);
// ordinary autodiff over it yields the R-lens backward
bool jlens_build_forward_lrp(const jlens_weights & m, int n_tokens, jlens_lrp_forward & out) {
    const auto & hp = m.hparams;

    const int    gsize    = hp.n_layer * 256;
    const int    n_tensors= hp.n_layer * 236;
    const size_t ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(gsize, true);
    struct ggml_init_params iparams = { ctx_size, nullptr, /*no_alloc=*/true };
    out.gctx = ggml_init(iparams);
    ggml_context * ctx = out.gctx;

    out.gf = ggml_new_graph_custom(ctx, gsize, /*grads=*/true);

    out.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(out.tokens, "jlens_tokens");
    ggml_set_input(out.tokens);

    out.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(out.pos, "jlens_pos");
    ggml_set_input(out.pos);

    out.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(out.mask, "jlens_mask");
    ggml_set_input(out.mask);

    ggml_tensor * inpL = ggml_get_rows(ctx, m.get("token_embd.weight"), out.tokens);

    // param-coverage hack (see jlens_build_forward)
    out.zero_pad = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(out.zero_pad, "jlens_zero_pad");
    ggml_set_param(out.zero_pad);
    inpL = ggml_add(ctx, inpL, out.zero_pad);

    out.leaves.resize(hp.n_layer);
    out.l_out.resize(hp.n_layer, nullptr);

    for (int il = 0; il < hp.n_layer; ++il) {
        jlens_lrp_leaves & lv = out.leaves[il];
        for (int k = 0; k < 4; ++k) {
            lv.rms_inv[k] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_tokens);
            char name[64];
            snprintf(name, sizeof(name), "jlens_lrp_%d_rms_%d", il, k);
            ggml_set_name(lv.rms_inv[k], name);
            ggml_set_input(lv.rms_inv[k]);
        }
        lv.sig_gate  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_ff, n_tokens);
        lv.silu_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_ff, n_tokens);
        lv.up        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_ff, n_tokens);
        char name[64];
        snprintf(name, sizeof(name), "jlens_lrp_%d_sig",  il); ggml_set_name(lv.sig_gate,  name); ggml_set_input(lv.sig_gate);
        snprintf(name, sizeof(name), "jlens_lrp_%d_silu", il); ggml_set_name(lv.silu_gate, name); ggml_set_input(lv.silu_gate);
        snprintf(name, sizeof(name), "jlens_lrp_%d_up",   il); ggml_set_name(lv.up,        name); ggml_set_input(lv.up);

        ggml_tensor * cur = glm_block(ctx, m, il, n_tokens, inpL, out.pos, out.mask,
                                      /*use_flash_attn=*/false, &lv, nullptr);

        snprintf(name, sizeof(name), "jlens_l_out_%d", il);
        ggml_set_name(cur, name);
        if (il == (int) hp.n_layer - 1) {
            cur->flags |= GGML_TENSOR_FLAG_LOSS; // caller supplies the grad_accs seed
        }
        ggml_build_forward_expand(out.gf, cur);
        out.l_out[il] = cur;
        inpL = cur;
    }

    return true;
}

bool jlens_build_upper_chunk(const jlens_weights & m, int slot_layer, int64_t a, int n_tokens, int64_t n_kv, int n_rows,
                             bool now_mode, jlens_upper_chunk & out, int top_layer) {
    const auto & hp = m.hparams;
    if (top_layer < 0) top_layer = (int) hp.n_layer - 1;
    const bool head = top_layer == hp.n_layer - 1;
    if (slot_layer < -1 || slot_layer >= top_layer || top_layer >= hp.n_layer || (head && (n_rows < 1 || n_rows > n_tokens)) ||
        (!head && now_mode) || a + n_tokens > n_kv) {
        fprintf(stderr, "jlens_build_upper_chunk: bad arguments\n");
        return false;
    }
    const int n_up = top_layer - slot_layer;
    const int64_t D = hp.n_embd_head, Hkv = hp.n_head_kv;

    const int    gsize    = n_up * 512 + 256;
    const int    n_tensors= n_up * 400 + 64;
    const size_t ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(gsize, true);
    struct ggml_init_params iparams = { ctx_size, nullptr, /*no_alloc=*/true };
    out.gctx = ggml_init(iparams);
    ggml_context * ctx = out.gctx;
    out.gf = ggml_new_graph_custom(ctx, gsize, /*grads=*/true);

    out.h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, n_tokens);
    ggml_set_name(out.h, "jlens_chunk_h");
    ggml_set_param(out.h);

    out.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(out.pos, "jlens_chunk_pos");
    ggml_set_input(out.pos);

    out.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
    ggml_set_name(out.mask, "jlens_chunk_mask");
    ggml_set_input(out.mask);

    out.one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(out.one, "jlens_chunk_one");

    out.k_kv.assign(n_up, nullptr);
    out.l_out.clear();
    out.v_kv.assign(n_up, nullptr);
    out.dk_in.assign(n_up, nullptr);
    out.dv_in.assign(n_up, nullptr);
    std::vector<jlens_chunk_kv> ckv(n_up);

    ggml_tensor * inpL = out.h;
    for (int u = 0; u < n_up; ++u) {
        const int il = slot_layer + 1 + u;
        char name[64];
        out.k_kv[u] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, n_kv);
        out.v_kv[u] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, n_kv);
        snprintf(name, sizeof(name), "jlens_chunk_k_%d", il); ggml_set_name(out.k_kv[u], name);
        snprintf(name, sizeof(name), "jlens_chunk_v_%d", il); ggml_set_name(out.v_kv[u], name);
        if (!now_mode) {
            ggml_set_param(out.k_kv[u]);
            ggml_set_param(out.v_kv[u]);
        }
        ckv[u].now  = now_mode;
        ckv[u].a    = a;
        ckv[u].k_kv = out.k_kv[u];
        ckv[u].v_kv = out.v_kv[u];
        ckv[u].one  = out.one;
        inpL = glm_block(ctx, m, il, n_tokens, inpL, out.pos, out.mask, /*use_flash_attn=*/false,
                         nullptr, nullptr, nullptr, nullptr, &ckv[u]);
        out.l_out.push_back(inpL);
    }

    ggml_tensor * loss;
    if (head) {
        // next-token loss on the first n_rows positions, as a sum (cross_entropy_loss is the row mean)
        ggml_tensor * x = n_rows == n_tokens ? inpL : ggml_view_2d(ctx, inpL, hp.n_embd, n_rows, inpL->nb[1], 0);
        ggml_tensor * logits = ggml_mul_mat(ctx, m.get("output.weight"), rms_norm_mul(ctx, x, m.get("output_norm.weight"), hp.norm_eps));
        out.labels = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, logits->ne[0], n_rows);
        ggml_set_name(out.labels, "jlens_chunk_labels");
        ggml_set_input(out.labels);
        loss = ggml_scale(ctx, ggml_cross_entropy_loss(ctx, logits, out.labels), (float) n_rows);
    } else {
        // gradient of the loss at the top_layer output, from the layer group above: add <g_top, h_top>
        out.g_top = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, n_tokens);
        ggml_set_name(out.g_top, "jlens_chunk_g_top");
        loss = ggml_sum(ctx, ggml_mul(ctx, inpL, out.g_top));
    }

    if (!now_mode) {
        // gradient that later chunks sent to this chunk's keys/values: add <dk_in, k> + <dv_in, v>
        for (int u = 0; u < n_up; ++u) {
            out.dk_in[u] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, n_tokens);
            out.dv_in[u] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, n_tokens);
            loss = ggml_add(ctx, loss, ggml_sum(ctx, ggml_mul(ctx, ckv[u].k_live, out.dk_in[u])));
            loss = ggml_add(ctx, loss, ggml_sum(ctx, ggml_mul(ctx, ckv[u].v_live, out.dv_in[u])));
        }
    }
    out.loss = loss;
    ggml_set_name(out.loss, "jlens_chunk_loss");
    ggml_set_output(out.loss);
    out.loss->flags |= GGML_TENSOR_FLAG_LOSS;
    ggml_build_forward_expand(out.gf, out.loss);
    return true;
}

bool jlens_build_chunk_forward(const jlens_weights & m, int64_t a, int n_tokens, int64_t n_kv, jlens_chunk_forward & out,
                               bool with_emb_add) {
    const auto & hp = m.hparams;
    if (a + n_tokens > n_kv) {
        fprintf(stderr, "jlens_build_chunk_forward: bad arguments\n");
        return false;
    }
    const int64_t D = hp.n_embd_head, Hkv = hp.n_head_kv;
    const int    gsize    = hp.n_layer * 96 + 64;
    const int    n_tensors= hp.n_layer * 96 + 64; // block ops + leaves + the caller's copy-out nodes
    const size_t ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + ggml_graph_overhead_custom(gsize, false);
    struct ggml_init_params iparams = { ctx_size, nullptr, /*no_alloc=*/true };
    out.gctx = ggml_init(iparams);
    ggml_context * ctx = out.gctx;
    out.gf = ggml_new_graph_custom(ctx, gsize, /*grads=*/false);

    out.tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(out.tokens);
    out.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(out.pos);
    out.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
    ggml_set_input(out.mask);
    out.one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    out.k_kv.assign(hp.n_layer, nullptr);
    out.v_kv.assign(hp.n_layer, nullptr);
    out.k_live.assign(hp.n_layer, nullptr);
    out.v_live.assign(hp.n_layer, nullptr);
    out.l_out.assign(hp.n_layer, nullptr);

    ggml_tensor * inpL = ggml_get_rows(ctx, m.get("token_embd.weight"), out.tokens);
    if (with_emb_add) {
        out.emb_add = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, n_tokens);
        inpL = ggml_add(ctx, inpL, out.emb_add);
    }
    out.emb = inpL;
    for (int il = 0; il < hp.n_layer; ++il) {
        out.k_kv[il] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, n_kv);
        out.v_kv[il] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, n_kv);
        jlens_chunk_kv ckv;
        ckv.a    = a;
        ckv.k_kv = out.k_kv[il];
        ckv.v_kv = out.v_kv[il];
        ckv.one  = out.one;
        inpL = glm_block(ctx, m, il, n_tokens, inpL, out.pos, out.mask, /*use_flash_attn=*/false,
                         nullptr, nullptr, nullptr, nullptr, &ckv);
        out.k_live[il] = ckv.k_live;
        out.v_live[il] = ckv.v_live;
        out.l_out[il]  = inpL;
    }
    return true;
}
