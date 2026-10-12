#include "models.h"

#include <algorithm>
#include <cstdio>
#include <vector>
#include "llama-memory-recurrent.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

//
// Kimi-K3 text model: hybrid KDA (linear) + MLA (full) attention, as in kimi-linear.
// Parts that kimi-linear does not have:
//   1. cross-layer residual attention  (attn_res_block_size)
//   2. latent MoE                      (routed experts run at n_expert_latent)
//   3. situ activation                 (replaces SwiGLU everywhere)
//   4. MLA output gate                 (sigmoid gate before o_proj)
//   5. full-rank KDA gate              (single ssm_g instead of ssm_g_a/ssm_g_b)
//

void llama_model_kimi_k3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q, false);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                hparams.n_embd_head_kda);
    ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND,        hparams.kda_gate_lower_bound, false);

    // the MLA cache holds the compressed latent
    // set it here too, as older GGUFs have no value_length key
    hparams.n_embd_head_v_full = hparams.n_lora_kv;

    // n_head_kv == 0 marks a KDA (recurrent) layer, as in kimi-linear
    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;
    }

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,  hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,       hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,        hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,         hparams.expert_gating_func);
    ml.get_key(LLM_KV_EXPERT_LATENT_LENGTH,       hparams.n_expert_latent, false);

    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,          hparams.attn_res_block_size);
    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,         hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA,  hparams.situ_linear_beta);

    switch (hparams.n_layer()) {
        case 93: type = LLM_TYPE_2_8T_A50B; break; // Kimi-K3
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_kimi_k3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_latent = hparams.n_expert_latent > 0 ? hparams.n_expert_latent : n_embd;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    if (hparams.attn_res_block_size > 0) {
        output_res_score = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_SCORE, "weight"), {n_embd}, 0);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, 0);

        if (hparams.attn_res_block_size > 0) {
            layer.attn_res_score = create_tensor(tn(LLM_TENSOR_ATTN_RES_SCORE, "weight", i), {n_embd}, 0);
            layer.ffn_res_score  = create_tensor(tn(LLM_TENSOR_FFN_RES_SCORE,  "weight", i), {n_embd}, 0);
        }

        const int64_t head_dim = hparams.n_embd_head_kda;
        const int64_t d_conv   = hparams.ssm_d_conv;
        const int64_t d_inner  = head_dim * n_head;

        if (hparams.is_recr(i)) {
            // conv1d may be stored 4D [d_conv, 1, d_inner, 1] or 3D (quantization drops the trailing 1)
            auto conv = [&](llm_tensor tid) {
                ggml_tensor * t = create_tensor(tn(tid, "weight", i), {d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
                return t ? t : create_tensor(tn(tid, "weight", i), {d_conv, 1, d_inner}, 0);
            };
            layer.ssm_q_conv = conv(LLM_TENSOR_SSM_CONV1D_Q);
            layer.ssm_k_conv = conv(LLM_TENSOR_SSM_CONV1D_K);
            layer.ssm_v_conv = conv(LLM_TENSOR_SSM_CONV1D_V);

            create_tensor_qkv(layer, i, n_embd, d_inner, d_inner, d_inner, 0);

            layer.ssm_f_a  = create_tensor(tn(LLM_TENSOR_SSM_F_A,  "weight", i), {n_embd, head_dim}, 0);
            layer.ssm_f_b  = create_tensor(tn(LLM_TENSOR_SSM_F_B,  "weight", i), {head_dim, d_inner}, 0);
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head}, 0);

            // K3's A_log is a plain 1-D [n_head] tensor (kimi-linear's is padded)
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN, i), {n_head}, 0);
            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, 0);

            // K3 uses a single full-rank gate instead of kimi-linear's g_a/g_b pair
            layer.ssm_g      = create_tensor(tn(LLM_TENSOR_SSM_G,    "weight", i), {n_embd, d_inner}, 0);
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {head_dim}, 0);
            layer.wo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {d_inner, n_embd}, 0);
        } else {
            const int64_t q_lora_rank      = hparams.n_lora_q;
            const int64_t kv_lora_rank     = hparams.n_lora_kv;
            const int64_t n_embd_head_k    = hparams.n_embd_head_k_mla();
            const int64_t n_embd_head_v    = hparams.n_embd_head_v_mla();
            const int64_t qk_rope_head_dim = hparams.n_rot();
            const int64_t qk_nope_head_dim = n_embd_head_k - qk_rope_head_dim;

            layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, TENSOR_NOT_REQUIRED);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);

            if (layer.attn_q_a_norm) {
                layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, 0);
                layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k}, 0);
            } else {
                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_head * n_embd_head_k}, 0);
            }

            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + qk_rope_head_dim}, 0);
            layer.wkv_b     = create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i),
                                            {kv_lora_rank, n_head * (qk_nope_head_dim + n_embd_head_v)},
                                            TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (!layer.wkv_b) {
                layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {qk_nope_head_dim, kv_lora_rank, n_head}, 0);
                layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v, n_head}, 0);
            }

            // K3: sigmoid output gate applied to the attention output before o_proj
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head * n_embd_head_v}, TENSOR_NOT_REQUIRED);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v, n_embd}, 0);
        }

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        } else {
            const int64_t n_ff_exp = hparams.n_ff_exp();

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);

            // routed experts live in the latent space
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd_latent, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd_latent, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd_latent, n_ff_exp, n_expert}, 0);

            if (hparams.n_expert_latent > 0) {
                layer.ffn_routed_down = create_tensor(tn(LLM_TENSOR_FFN_ROUTED_DOWN, "weight", i), {n_embd, n_embd_latent}, 0);
                layer.ffn_routed_up   = create_tensor(tn(LLM_TENSOR_FFN_ROUTED_UP,   "weight", i), {n_embd_latent, n_embd}, 0);
                layer.ffn_routed_norm = create_tensor(tn(LLM_TENSOR_FFN_ROUTED_NORM, "weight", i), {n_embd_latent}, TENSOR_NOT_REQUIRED);
            }

            // shared experts stay at n_embd, width = moe_intermediate_size * n_expert_shared
            const int64_t n_ff_shexp = n_ff_exp * (hparams.n_expert_shared > 0 ? hparams.n_expert_shared : 1);
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_kimi_k3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)
// linear_beta <= 0 disables the transform on the up branch
static ggml_tensor * kimi_k3_situ(ggml_context * ctx0, ggml_tensor * gate, ggml_tensor * up,
                                  float beta, float linear_beta) {
    return ggml_situ_split(ctx0, gate, up, beta, linear_beta);
}

// TEMP experiment: LLAMA_K3_PRED_DUMP=<file> appends router selection scores (sigmoid(logits) + bias) from a
// CPU custom op: int32 tag, layer, n_expert, n_tokens, then n_tokens x n_expert floats. tag 0 = actual,
// 1 = predicted from the pre-attention stream through the ffn-side mix, 2 = from the attention input
static const char * k3_pred_dump_path() {
    static const char * path = getenv("LLAMA_K3_PRED_DUMP");
    return path;
}

static void k3_pred_dump_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * ud) {
    GGML_UNUSED(dst);
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    static FILE * f = fopen(k3_pred_dump_path(), "ab");
    const int32_t hdr[4] = { (int32_t) ((intptr_t) ud / 1000), (int32_t) ((intptr_t) ud % 1000),
                             (int32_t) a->ne[0], (int32_t) a->ne[1] };
    fwrite(hdr, sizeof(hdr), 1, f);
    for (int64_t t = 0; t < a->ne[1]; ++t) {
        fwrite((const char *) a->data + t*a->nb[1], sizeof(float), a->ne[0], f);
    }
    fflush(f);
}

static void k3_pred_dump(ggml_context * ctx0, ggml_cgraph * gf, ggml_tensor * logits, ggml_tensor * probs_b,
                         int tag, int il) {
    ggml_tensor * s = ggml_add(ctx0, ggml_sigmoid(ctx0, logits), probs_b);
    ggml_build_forward_expand(gf, ggml_map_custom1(ctx0, s, k3_pred_dump_op, 1, (void *) (intptr_t) (tag*1000 + il)));
}

// expert prefetch during attention (decode, LLAMA_MOE_PREFETCH=<n experts>): the layer's router is run on the
// pre-attention stream on the GPU, attention is deferred, and a CPU op reads the predicted experts into the L3
// of the cores that will compute them while the GPU runs attention (see ggml_cpu_moe_prefetch)
struct k3_moe_pf_cfg {
    int n;      // experts to prefetch per layer
    int banked; // also on the layers that restart the residual stream (poorly predicted)
    int gpu_k;  // the GPU share takes the first gpu_k predicted ids < gpu_e1: not prefetched
    int gpu_e1;
};

static const k3_moe_pf_cfg & k3_moe_pf(int n_expert) {
    static k3_moe_pf_cfg cfg = [n_expert] {
        k3_moe_pf_cfg c;
        const float frac = getenv("LLAMA_MOE_GPU_FRAC") ? atof(getenv("LLAMA_MOE_GPU_FRAC")) : 0.0f;
        c.n      = getenv("LLAMA_MOE_PREFETCH") ? atoi(getenv("LLAMA_MOE_PREFETCH")) : 0;
        c.banked = getenv("LLAMA_MOE_PREFETCH_BANKED") ? atoi(getenv("LLAMA_MOE_PREFETCH_BANKED")) : 0;
        c.gpu_k  = frac > 0.0f ? (getenv("LLAMA_MOE_GPU_K") ? atoi(getenv("LLAMA_MOE_GPU_K")) : 3) : 0;
        c.gpu_e1 = (int) (frac * n_expert);
        return c;
    }();
    return cfg;
}

static void k3_moe_prefetch_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    const k3_moe_pf_cfg * cfg = (const k3_moe_pf_cfg *) ud;
    const ggml_tensor * ids = dst->src[0]; // predicted top-k, descending

    int32_t sel[32];
    int n_sel = 0;
    int n_gpu = 0;
    for (int64_t i = 0; i < ids->ne[0] && n_sel < std::min(cfg->n, 32); ++i) {
        const int32_t e = ((const int32_t *) ids->data)[i];
        if (n_gpu < cfg->gpu_k && e < cfg->gpu_e1) {
            n_gpu++;
            continue;
        }
        sel[n_sel++] = e;
    }
    const ggml_tensor * as[3] = { dst->src[1], dst->src[2], dst->src[3] };
    ggml_cpu_moe_prefetch(as, 3, sel, n_sel, ith, nth);
}

//
// cross-layer residual attention
//

// layout is [n_embd, n_ckpt, n_tokens]: rms_norm reduces over ne0, dsv4_hc_pre over ne1
// append the new checkpoint, do not re-fold the whole chain
void llama_model_kimi_k3::graph::res_push(ggml_tensor * cur, int64_t n_embd, int64_t n_tokens) {
    ggml_tensor * ckpt = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    resi_stack = resi_stack ? ggml_concat(ctx0, resi_stack, ckpt, 1) : ckpt;
}

ggml_tensor * llama_model_kimi_k3::graph::res_mix(ggml_tensor * cur, ggml_tensor * score_w,
                                                  int64_t n_tokens, int il) {
    if (!resi_stack) {
        return cur; // layer 0: nothing banked yet
    }

    const float eps = hparams.f_norm_rms_eps;

    // score the banked checkpoints and the current stream together: [n_embd, n_ckpt + 1, n_tokens]
    // note: the scores use the normalized values, but the sum below uses the raw ones
    ggml_tensor * src = ggml_concat(ctx0, resi_stack, ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, n_tokens), 1);
    const int n_src = (int) src->ne[1];

    ggml_tensor * scores = ggml_rms_norm(ctx0, src, eps);
    scores = ggml_mul(ctx0, scores, score_w);
    scores = ggml_sum_rows(ctx0, scores);                          // [1, n_src, n_tokens]
    scores = ggml_reshape_2d(ctx0, scores, n_src, n_tokens);

    ggml_tensor * probs = ggml_soft_max(ctx0, scores);            // over ne0 = n_src
    cb(probs, "res_probs", il);

    ggml_tensor * out = ggml_dsv4_hc_pre(ctx0, src, probs);
    return out;
}

llama_model_kimi_k3::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "inp_embd", -1);

    // K3 MLA is nope-only, so there is no position input

    auto * inp_kv      = !hparams.is_mla() ? build_inp_mem_hybrid()   : nullptr;
    auto * inp_k       =  hparams.is_mla() ? build_inp_mem_hybrid_k() : nullptr;
    auto * inp_rs      =  hparams.is_mla() ? inp_k->get_recr() : inp_kv->get_recr();
    auto * inp_attn_kv = !hparams.is_mla() ? inp_kv->get_attn() : nullptr;
    auto * inp_attn_k  =  hparams.is_mla() ? inp_k->get_attn()  : nullptr;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_head_kda = hparams.n_head();
    const int64_t head_dim   = hparams.n_embd_head_kda;
    const int64_t d_conv     = hparams.ssm_d_conv;
    const int64_t d_inner    = n_head_kda * head_dim;
    const int64_t n_seqs     = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla   = hparams.n_embd_head_v_mla();
    const int64_t kv_lora_rank        = hparams.n_lora_kv;
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    const float   kq_scale_mla        = 1.0f / sqrtf((float) n_embd_head_k_mla);

    const uint32_t res_bs        = hparams.attn_res_block_size;
    const bool     use_attn_res  = res_bs > 0;
    const int64_t  n_embd_latent = hparams.n_expert_latent > 0 ? hparams.n_expert_latent : n_embd;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // the residual stream, banked on checkpoint layers and then restarted
        // from the attention output alone
        ggml_tensor * prefix_sum = inpL;

        cur = use_attn_res ? res_mix(prefix_sum, layer.attn_res_score, n_tokens, il)
                           : prefix_sum;

        // the stream as this layer's attention reads it (pre-norm mixture), which is
        // what draft models (DFlash) are trained on as "the output of layer il-1"
        res->t_layer_inp[il] = cur;

        if (k3_pred_dump_path() && use_attn_res && layer.ffn_gate_inp) {
            for (int tag : {1, 2}) {
                ggml_tensor * x = tag == 1 ? res_mix(prefix_sum, layer.ffn_res_score, n_tokens, il) : cur;
                x = build_norm(x, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
                k3_pred_dump(ctx0, gf, ggml_mul_mat(ctx0, layer.ffn_gate_inp, x), layer.ffn_exp_probs_b, tag, il);
            }
        }

        // expert prefetch: predicted ids before attention (the stack is still the one the ffn-side mix will see)
        const k3_moe_pf_cfg & pf_cfg = k3_moe_pf(hparams.n_expert);
        ggml_tensor * pf_ids = nullptr;
        if (pf_cfg.n > 0 && n_tokens == 1 && use_attn_res && layer.ffn_gate_inp && layer.ffn_up_exps &&
                layer.ffn_gate_exps && layer.ffn_down_exps && layer.ffn_up_exps->buffer &&
                ggml_backend_buffer_is_host(layer.ffn_up_exps->buffer) &&
                (pf_cfg.banked || (uint32_t) il % res_bs != 0)) {
            ggml_tensor * x = res_mix(prefix_sum, layer.ffn_res_score, n_tokens, il);
            x = build_norm(x, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
            x = ggml_add(ctx0, ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, layer.ffn_gate_inp, x)), layer.ffn_exp_probs_b);
            pf_ids = ggml_top_k(ctx0, x, hparams.n_expert_used());
            cb(pf_ids, "ffn_moe_pf_ids", il);
            ggml_build_forward_expand(gf, pf_ids);
        }
        const int pf_node0 = ggml_graph_n_nodes(gf);

        bool banked = false;
        if (use_attn_res && (uint32_t) il % res_bs == 0) {
            res_push(prefix_sum, n_embd, n_tokens);  // banks the RAW layer input, not `cur`
            banked = true;
        }

        cur = build_norm(cur, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_kda_layer(cur, layer, inp_rs, d_conv, head_dim, n_head_kda,
                                  d_inner, n_seq_tokens, n_seqs, il);
        } else {
            cur = build_mla_layer(cur, layer, inp_attn_k, inp_attn_kv,
                                  n_embd_head_k_mla, n_embd_head_v_mla, kv_lora_rank,
                                  n_embd_head_qk_rope, n_embd_head_qk_nope, kq_scale_mla, il);
        }

        // defer the attention split so that it runs on the GPU while the CPU split below prefetches
        if (pf_ids) {
            ggml_build_forward_expand(gf, cur);
            for (int i = pf_node0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_graph_node(gf, i)->flags |= GGML_TENSOR_FLAG_DEFER;
            }
            ggml_tensor * args[4] = { pf_ids, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps };
            ggml_tensor * pf = ggml_custom_4d(ctx0, GGML_TYPE_I32, 1, 1, 1, 1, args, 4, k3_moe_prefetch_op,
                                              GGML_N_TASKS_MAX, const_cast<k3_moe_pf_cfg *>(&pf_cfg));
            cb(pf, "ffn_moe_prefetch", il);
            ggml_build_forward_expand(gf, pf);
        }

        prefix_sum = banked ? cur : ggml_add(ctx0, prefix_sum, cur);
        cb(prefix_sum, "prefix_sum_attn", il);

        cur = use_attn_res ? res_mix(prefix_sum, layer.ffn_res_score, n_tokens, il)
                           : prefix_sum;

        cur = build_norm(cur, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            ggml_tensor * g = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
            ggml_tensor * u = ggml_mul_mat(ctx0, layer.ffn_up,   cur);
            cur = kimi_k3_situ(ctx0, g, u, hparams.situ_beta, hparams.situ_linear_beta);
            cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);
            cb(cur, "ffn_out", il);
        } else {
            cur = build_latent_moe(cur, layer, n_embd_latent, il);
        }

        prefix_sum = ggml_add(ctx0, prefix_sum, cur);
        prefix_sum = build_cvec(prefix_sum, il);
        cb(prefix_sum, "l_out", il);

        inpL = prefix_sum;
    }

    cur = inpL;

    // final mix, then narrow to the output tokens
    if (use_attn_res) {
        cur = res_mix(cur, model.output_res_score, n_tokens, -1);
    }
    res->t_layer_inp[n_layer] = cur;
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

//
// KDA layer
//

// causal conv1d over one of Q/K/V. `qkv` selects which third of the conv state to use
static ggml_tensor * kimi_k3_conv1d(ggml_cgraph * gf, ggml_context * ctx0,
                                    ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
                                    int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                                    int64_t d_conv, int64_t head_dim, int64_t n_head,
                                    int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head,
                                    int64_t mem_size, int64_t K_rs) {
    const int64_t d_inner         = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total  = 3 * conv_state_size;

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1)   * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    ggml_tensor * x_3d   = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    // group s holds the conv window s tokens back.
    // [TAG_RECURRENT_ROLLBACK_SPLITS]: the last K_rs tokens must share one ubatch.
    for (int64_t s = 0; s < K_rs; ++s) {
        const int64_t s_idx = std::max<int64_t>(0, n_seq_tokens - s);
        ggml_tensor * conv_x_s = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
            conv_x->nb[1], conv_x->nb[2], s_idx * conv_x->nb[0]);
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, conv_x_s,
                ggml_view_3d(ctx0, conv_states_all, d_conv - 1, d_inner, n_seqs,
                    (d_conv - 1)   * ggml_element_size(conv_states_all),
                    n_embd_r_total * ggml_element_size(conv_states_all),
                    ((s * mem_size + kv_head) * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));
    }

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

ggml_tensor * llama_model_kimi_k3::graph::build_kda_layer(
        ggml_tensor * cur, const llama_layer & layer, llm_graph_input_rs * inp_rs,
        int64_t d_conv, int64_t head_dim, int64_t n_head_kda,
        int64_t d_inner, int64_t n_seq_tokens, int64_t n_seqs, int il) {

    const auto * mctx_cur = inp_rs->mctx;
    const auto   kv_head  = mctx_cur->get_head();

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * conv_state_all  = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

    const int64_t mem_size = mctx_cur->get_size();
    const int64_t K_rs     = (int64_t) cparams.n_rs_seq + 1;

    ggml_tensor * Qcur = kimi_k3_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head, mem_size, K_rs);
    ggml_tensor * Kcur = kimi_k3_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head, mem_size, K_rs);
    ggml_tensor * Vcur = kimi_k3_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head, mem_size, K_rs);
    cb(Qcur, "kda_q_conv", il);
    cb(Kcur, "kda_k_conv", il);
    cb(Vcur, "kda_v_conv", il);

    // gate_lower_bound is not a clamp - when set, it swaps the decay gate activation:
    //   unset (kimi-linear):  g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
    //   set   (K3, -5.0):     g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
    // ssm_a holds -exp(A_log) (folded at conversion time), so exp(A_log) == -ssm_a
    ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
    ggml_tensor * g1  = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
    g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);

    ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head_kda, 1);

    if (hparams.kda_gate_lower_bound > -INFINITY) {
        g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, n_tokens);
        g1 = ggml_mul(ctx0, g1, A);                                    // -exp(A_log) * (...)
        g1 = ggml_sigmoid(ctx0, ggml_scale(ctx0, g1, -1.0f));
        g1 = ggml_scale(ctx0, g1, hparams.kda_gate_lower_bound);
    } else {
        g1 = ggml_softplus(ctx0, g1);
        g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, n_tokens);
        g1 = ggml_mul(ctx0, g1, A);
    }
    cb(g1, "kda_g1", il);

    g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head_kda, n_seq_tokens, n_seqs);

    ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, n_head_kda, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "kda_beta", il);

    ggml_tensor * cur_3d = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head_kda, n_seqs);

    const float eps_norm = hparams.f_norm_rms_eps;
    Qcur = build_gdn_l2_norm(ctx0, Qcur, eps_norm);
    Kcur = build_gdn_l2_norm(ctx0, Kcur, eps_norm);

    ggml_tensor * output = build_recurrent_attn(inp_rs, ssm_states_all, Qcur, Kcur, Vcur, g1, beta, state, il);
    output = ggml_cont(ctx0, output);
    cb(output, "kda_scan_out", il);

    // K3: single full-rank gate (kimi-linear factors this as g_b(g_a(x)))
    ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur_3d, cur_3d->ne[0], n_seq_tokens * n_seqs);
    ggml_tensor * g2     = ggml_mul_mat(ctx0, layer.ssm_g, cur_2d);
    g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head_kda, n_seq_tokens * n_seqs);

    ggml_tensor * o      = ggml_reshape_3d(ctx0, output, head_dim, n_head_kda, n_seq_tokens * n_seqs);
    ggml_tensor * normed = build_norm(o, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
    cb(g2, "kda_g2", il);
    cb(normed, "kda_normed", il);
    ggml_tensor * gated  = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, g2));

    gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
    cur   = ggml_mul_mat(ctx0, layer.wo, gated);
    cb(cur, "kda_out", il);

    return cur;
}

//
// MLA layer (nope-only, with K3's sigmoid output gate)
//

ggml_tensor * llama_model_kimi_k3::graph::build_mla_layer(
        ggml_tensor * cur, const llama_layer & layer,
        llm_graph_input_attn_k * inp_attn_k, llm_graph_input_attn_kv * inp_attn_kv,
        int64_t n_embd_head_k_mla, int64_t n_embd_head_v_mla, int64_t kv_lora_rank,
        int64_t n_embd_head_qk_rope, int64_t n_embd_head_qk_nope, float kq_scale, int il) {

    ggml_tensor * inp_gate = cur; // the output gate reads the *normed* layer input

    ggml_tensor * Qcur;
    if (layer.wq_a) {
        Qcur = ggml_mul_mat(ctx0, layer.wq_a, cur);
        Qcur = build_norm(Qcur, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        Qcur = ggml_mul_mat(ctx0, layer.wq_b, Qcur);
    } else {
        Qcur = ggml_mul_mat(ctx0, layer.wq, cur);
    }

    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);

    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    // no RoPE: mla_use_nope is asserted at conversion time
    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * out;
    if (layer.wk_b && layer.wv_b) {
        ggml_tensor * q_nope = ggml_view_3d(ctx0, Qcur, n_embd_head_qk_nope, n_head, n_tokens,
            ggml_row_size(Qcur->type, n_embd_head_k_mla),
            ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head, 0);
        ggml_tensor * q_pe = ggml_view_3d(ctx0, Qcur, n_embd_head_qk_rope, n_head, n_tokens,
            ggml_row_size(Qcur->type, n_embd_head_k_mla),
            ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head,
            ggml_row_size(Qcur->type, n_embd_head_qk_nope));

        q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
        q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

        ggml_tensor * Q = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
        ggml_tensor * kv_cmpr_3d = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
        ggml_tensor * K = ggml_concat(ctx0, kv_cmpr_3d, k_pe, 0);
        ggml_tensor * V = kv_cmpr_3d;

        // wo == NULL: the output projection is applied after the gate below
        out = build_attn(inp_attn_k, nullptr, NULL, nullptr, Q, K, V, nullptr, nullptr, layer.wv_b, kq_scale, il);
    } else {
        ggml_tensor * Q = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k_mla, n_head, n_tokens);
        ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_cmpr);
        const int64_t kv_per_head = n_embd_head_qk_nope + n_embd_head_v_mla;

        ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
            ggml_row_size(kv->type, kv_per_head), ggml_row_size(kv->type, kv_per_head * n_head), 0);
        ggml_tensor * V = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, n_embd_head_v_mla, n_head, n_tokens,
            ggml_row_size(kv->type, kv_per_head), ggml_row_size(kv->type, kv_per_head * n_head),
            ggml_row_size(kv->type, n_embd_head_qk_nope)));

        ggml_tensor * k_pe_t = ggml_new_tensor_3d(ctx0, k_pe->type, n_embd_head_qk_rope, n_head, n_tokens);
        ggml_tensor * K = ggml_concat(ctx0, ggml_repeat(ctx0, k_pe, k_pe_t), k_nope, 0);

        out = build_attn(inp_attn_kv, nullptr, NULL, nullptr, Q, K, V, nullptr, nullptr, nullptr, kq_scale, il);
    }

    // K3: attn_output *= sigmoid(g_proj(x)), then o_proj
    if (layer.wqkv_gate) {
        ggml_tensor * g = ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, layer.wqkv_gate, inp_gate));
        out = ggml_mul(ctx0, out, g);
        cb(out, "mla_gated", il);
    }

    out = ggml_mul_mat(ctx0, layer.wo, out);
    cb(out, "mla_out", il);

    return out;
}

//
// latent MoE: down-project, run the routed experts in the latent space, norm, up-project;
// shared experts stay at n_embd and read the un-projected input.
//

// TEMP experiment: LLAMA_K3_XPROJ=<gguf> projects the routed experts' latent input onto a per-layer
// subspace, x -> P (P^T x), with blk.N.xproj = P^T [3584 -> r] and blk.N.xprojt = P [r -> 3584].
// equivalent to storing the gate/up experts as W P (rank-r inputs); used to measure the quality cost
struct k3_xproj_store {
    std::vector<ggml_tensor *> p;
    std::vector<ggml_tensor *> pt;
};

static const k3_xproj_store & k3_xproj() {
    static k3_xproj_store s;
    static bool init = false;
    if (init) {
        return s;
    }
    init = true;
    const char * path = getenv("LLAMA_K3_XPROJ");
    if (!path) {
        return s;
    }
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    GGML_ASSERT(g && "LLAMA_K3_XPROJ: cannot read file");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(meta, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    FILE * f = fopen(path, "rb");
    GGML_ASSERT(f);
    const size_t data_off = gguf_get_data_offset(g);
    s.p.assign(LLAMA_MAX_LAYERS, nullptr);
    s.pt.assign(LLAMA_MAX_LAYERS, nullptr);
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(meta, name);
        GGML_ASSERT(fseek(f, (long) (data_off + gguf_get_tensor_offset(g, i)), SEEK_SET) == 0);
        GGML_ASSERT(fread(t->data, 1, ggml_nbytes(t), f) == ggml_nbytes(t));
        int il = -1;
        char kind[16] = {0};
        if (sscanf(name, "blk.%d.%15s", &il, kind) == 2 && il >= 0 && il < LLAMA_MAX_LAYERS) {
            if (strcmp(kind, "xproj") == 0)  { s.p[il]  = t; }
            if (strcmp(kind, "xprojt") == 0) { s.pt[il] = t; }
        }
    }
    const int64_t n_loaded = gguf_get_n_tensors(g);
    fclose(f);
    gguf_free(g);
    LLAMA_LOG_INFO("%s: loaded %lld tensors from %s\n", __func__, (long long) n_loaded, path);
    return s;
}

ggml_tensor * llama_model_kimi_k3::graph::build_latent_moe(
        ggml_tensor * cur, const llama_layer & layer, int64_t n_embd_latent, int il) {

    ggml_tensor * identity = cur;

    // the shared expert only needs the layer input: build it first and let it run on the GPU
    // while the CPU computes the routed experts (see GGML_TENSOR_FLAG_DEFER)
    ggml_tensor * sh = nullptr;
    if (layer.ffn_gate_shexp) {
        ggml_tensor * g = ggml_mul_mat(ctx0, layer.ffn_gate_shexp, identity);
        ggml_tensor * u = ggml_mul_mat(ctx0, layer.ffn_up_shexp,   identity);
        ggml_tensor * a = kimi_k3_situ(ctx0, g, u, hparams.situ_beta, hparams.situ_linear_beta);
        sh = ggml_mul_mat(ctx0, layer.ffn_down_shexp, a);
        cb(sh, "ffn_shexp", il);
        for (ggml_tensor * t : {g, u, a, sh}) {
            t->flags |= GGML_TENSOR_FLAG_DEFER;
        }
        moe_defer = sh;
    }

    ggml_tensor * routed_in = layer.ffn_routed_down
        ? ggml_mul_mat(ctx0, layer.ffn_routed_down, cur)
        : cur;

    {
        const k3_xproj_store & xp = k3_xproj();
        if (!xp.p.empty() && xp.p[il] && xp.pt[il]) {
            routed_in = ggml_mul_mat(ctx0, xp.pt[il], ggml_mul_mat(ctx0, xp.p[il], routed_in));
            cb(routed_in, "ffn_routed_in_proj", il);
        }
    }

    // the router scores the full-width input while the experts take the latent one,
    // so the logits are computed here and passed to build_moe_ffn
    ggml_tensor * logits = ggml_mul_mat(ctx0, layer.ffn_gate_inp, identity);
    cb(logits, "ffn_moe_logits", il);
    if (k3_pred_dump_path()) {
        k3_pred_dump(ctx0, gf, logits, layer.ffn_exp_probs_b, 0, il);
    }

    // keep the latent projection before the deferred shared expert, so that split is followed by the experts
    ggml_build_forward_expand(gf, routed_in);

    // optional GPU share of the routed experts: LLAMA_MOE_GPU_FRAC = fraction of expert ids (from 0) the GPU may take,
    // LLAMA_MOE_GPU_K = max experts on the GPU at one token, LLAMA_MOE_GPU_KB at 2-4 tokens (speculative verify;
    // default K). Needs HMM for the host-resident expert weights
    static const float gpu_frac = getenv("LLAMA_MOE_GPU_FRAC") ? atof(getenv("LLAMA_MOE_GPU_FRAC")) : 0.0f;
    static const int   gpu_k    = getenv("LLAMA_MOE_GPU_K")    ? atoi(getenv("LLAMA_MOE_GPU_K"))    : 3;
    static const int   gpu_kb   = getenv("LLAMA_MOE_GPU_KB")   ? atoi(getenv("LLAMA_MOE_GPU_KB"))   : gpu_k;
    if (gpu_frac > 0.0f && layer.ffn_up_exps && layer.ffn_up_exps->buffer && ggml_backend_buffer_is_host(layer.ffn_up_exps->buffer)) {
        moe_gpu_k  = n_tokens == 1 ? gpu_k : gpu_kb;
        moe_gpu_e1 = (int) (gpu_frac * hparams.n_expert);
    }

    ggml_tensor * moe_out = build_moe_ffn(routed_in,
        nullptr, // gate_inp unused: the logits above are passed instead
        layer.ffn_up_exps,
        layer.ffn_gate_exps,
        layer.ffn_down_exps,
        layer.ffn_exp_probs_b,
        hparams.n_expert,
        hparams.n_expert_used(),
        LLM_FFN_SITU, hparams.expert_weights_norm,
        hparams.expert_weights_scale,
        (llama_expert_gating_func_type) hparams.expert_gating_func,
        il,
        logits);
    cb(moe_out, "ffn_moe_out", il);

    if (layer.ffn_routed_norm) {
        moe_out = build_norm(moe_out, layer.ffn_routed_norm, NULL, LLM_NORM_RMS, il);
    }
    if (layer.ffn_routed_up) {
        moe_out = ggml_mul_mat(ctx0, layer.ffn_routed_up, moe_out);
    }
    GGML_UNUSED(n_embd_latent);

    if (sh) {
        moe_out = ggml_add(ctx0, moe_out, sh);
    }

    cb(moe_out, "ffn_out", il);
    return moe_out;
}
