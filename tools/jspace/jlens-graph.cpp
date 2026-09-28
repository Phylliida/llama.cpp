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

// one GLM-4 dense block: post-norm attention + swiglu MLP with residuals
static ggml_tensor * glm_block(ggml_context * ctx, const jlens_weights & m, int il, int n_tokens,
                               ggml_tensor * inpL, ggml_tensor * pos, ggml_tensor * mask,
                               bool use_flash_attn) {
    const auto & hp = m.hparams;
    const int64_t n_embd      = hp.n_embd;
    const int64_t n_head      = hp.n_head;
    const int64_t n_head_kv   = hp.n_head_kv;
    const int64_t n_embd_head = hp.n_embd_head;
    const int64_t n_ff        = hp.n_ff;
    const float   kq_scale    = 1.0f / sqrtf((float) n_embd_head);

    ggml_tensor * inpSA = inpL;

    // pre-attention norm
    ggml_tensor * cur = rms_norm_mul(ctx, inpL, m.blk("blk.%d.attn_norm.weight", il), hp.norm_eps);

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

    // GQA: ggml_mul_mat broadcasts a's head dims (b head i maps to a head i / (Hb/Ha),
    // contiguous groups) and the MUL_MAT backward handles the broadcast, so no manual
    // kv-head expansion is needed.
    ggml_tensor * kqv;
    if (use_flash_attn) {
        // forward-only path: flash attention keeps the score matrix off VRAM.
        // layout [n_embd_head, n_tokens, n_head], k/v cast to f16 (CUDA fattn),
        // GQA (n_head_kv < n_head) handled by the kernel. No backward support.
        ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        kp = ggml_cast(ctx, kp, GGML_TYPE_F16);
        vp = ggml_cast(ctx, vp, GGML_TYPE_F16);
        qp = ggml_cast(ctx, qp, GGML_TYPE_F16);
        ggml_tensor * fmask = ggml_cast(ctx, mask, GGML_TYPE_F16); // flash attn requires f16 mask
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, qp, kp, vp, fmask, kq_scale,
                                               /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        kqv = ggml_cont(ctx, ggml_permute(ctx, fa, 0, 2, 1, 3));       // [n_embd_head, n_head, n_q]
        kqv = ggml_reshape_2d(ctx, kqv, n_embd, n_tokens);
    } else {
        // to [n_embd_head, n_tokens, n_head] layout (heads last) for batched attention matmuls
        ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

        // attention scores [n_kv, n_q, n_head] (k broadcast 2 -> 48 heads)
        ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);
        kq = ggml_soft_max_ext(ctx, kq, mask, kq_scale, /*max_bias=*/0.0f);

        // weighted values: [n_embd_head, n_q, n_head]
        // nb: build vt with a single permute+cont from v. A cont->transpose->cont chain
        // (or cont+mul_mat on a transposed view) makes the backward pass hand a
        // non-contiguous grad view to a CONT node, which ggml asserts against.
        ggml_tensor * vt  = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [n_kv, n_embd_head, n_head_kv]
        kqv = ggml_mul_mat(ctx, vt, kq);                        // [n_embd_head, n_q, n_head]

        // merge heads -> [n_embd, n_q]
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));       // [n_embd_head, n_head, n_q]
        kqv = ggml_reshape_2d(ctx, kqv, n_embd, n_tokens);
    }

    cur = ggml_mul_mat(ctx, m.blk("blk.%d.attn_output.weight", il), kqv);

    // post-attention norm, residual
    cur = rms_norm_mul(ctx, cur, m.blk("blk.%d.post_attention_norm.weight", il), hp.norm_eps);
    ggml_tensor * ffn_inp = ggml_add(ctx, cur, inpSA);

    // pre-MLP norm
    cur = rms_norm_mul(ctx, ffn_inp, m.blk("blk.%d.ffn_norm.weight", il), hp.norm_eps);

    // combined gate|up projection [2*n_ff, n]; split for swiglu_split (backward needs split form)
    ggml_tensor * up_out = ggml_mul_mat(ctx, m.blk("blk.%d.ffn_up.weight", il), cur);
    ggml_tensor * gate   = ggml_cont(ctx, ggml_view_2d(ctx, up_out, n_ff, n_tokens,
                                    up_out->nb[1], 0));
    ggml_tensor * up     = ggml_cont(ctx, ggml_view_2d(ctx, up_out, n_ff, n_tokens,
                                    up_out->nb[1], n_ff * up_out->nb[0]));
    cur = ggml_swiglu_split(ctx, gate, up);                          // silu(gate) * up, [n_ff, n]
    cur = ggml_mul_mat(ctx, m.blk("blk.%d.ffn_down.weight", il), cur);

    // post-MLP norm, residual
    cur = rms_norm_mul(ctx, cur, m.blk("blk.%d.post_ffw_norm.weight", il), hp.norm_eps);
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
                         int perturb_layer, bool use_flash_attn) {
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

    for (int il = 0; il < hp.n_layer; ++il) {
        ggml_tensor * cur = glm_block(ctx, m, il, n_tokens, inpL, out.pos, out.mask, use_flash_attn);

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
        if (with_grad_flags && il == perturb_layer) {
            // additive probe input for finite-difference validation of the backward pass
            out.perturb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, n_tokens);
            ggml_set_name(out.perturb, "jlens_perturb");
            cur = ggml_add(ctx, cur, out.perturb);
        }
        inpL = cur;
    }

    // full-model head: rms norm + output projection, [n_vocab, n_tokens]
    ggml_tensor * normed = rms_norm_mul(ctx, inpL, m.get("output_norm.weight"), hp.norm_eps);
    out.logits = ggml_mul_mat(ctx, m.get("output.weight"), normed);
    ggml_set_name(out.logits, "jlens_logits");
    ggml_build_forward_expand(out.gf, out.logits);

    ggml_set_output(out.logits);
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
