#pragma once

// J-lens tooling: load a GLM-4 dense GGUF into a ggml CUDA buffer and build
// custom forward graphs that expose every block's residual-stream output.
//
// The graphs are built by hand (not via llama.cpp's graph builders) so that
//   - every op has a working ggml backward path (explicit attention by default;
//     the flash-attention variant is forward-only, used by the server readout)
//   - block outputs can be flagged for gradient accumulation (J-lens VJP fitting)

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct jlens_hparams {
    int64_t n_embd      = 0;
    int64_t n_layer     = 0;
    int64_t n_head      = 0;
    int64_t n_head_kv   = 0;
    int64_t n_embd_head = 0;
    int64_t n_ff        = 0;
    int64_t n_rot       = 0;
    int64_t n_vocab     = 0;
    int64_t n_ctx_train = 0;
    float   rope_freq_base = 10000.0f;
    float   norm_eps       = 1e-5f;
};

// minimal weight-access interface needed by the graph builders, so the same
// graph code can run against either our own GGUF loader (jlens_model) or a
// model already loaded inside llama-server (adapter over its tensors_by_name)
struct jlens_weights {
    jlens_hparams hparams;

    virtual ~jlens_weights() = default;
    virtual ggml_tensor * get(const std::string & name) const = 0;

    ggml_tensor * blk(const char * fmt, int il) const; // fmt like "blk.%d.attn_q.weight"
};

struct jlens_model : public jlens_weights {
    gguf_context *             gguf = nullptr;
    ggml_context *             wctx = nullptr; // owns weight tensors (no_alloc)
    ggml_backend_buffer_t      wbuf = nullptr; // device buffer holding weight data
    std::vector<uint8_t *>     mmap_ptrs;      // for cleanup
    void *                     file_map  = nullptr;
    size_t                     file_size = 0;

    // name -> tensor, e.g. "blk.0.attn_q.weight"
    std::unordered_map<std::string, ggml_tensor *> tensors;

    // tokenizer.ggml.tokens for id -> string in readout
    std::vector<std::string> vocab_tokens;

    ggml_tensor * get(const std::string & name) const override;
};

// load GGUF, create tensors in the given backend buffer type (e.g. CUDA), upload data
bool jlens_model_load(const char * path, ggml_backend_buffer_type_t buft, jlens_model & out);
void jlens_model_free(jlens_model & m);

struct jlens_forward {
    ggml_context * gctx = nullptr; // graph tensor context (no_alloc)
    ggml_cgraph  * gf   = nullptr;

    ggml_tensor * tokens = nullptr; // i32 [n_tokens] input
    ggml_tensor * pos    = nullptr; // i32 [n_tokens] input
    ggml_tensor * mask   = nullptr; // f32 [n_tokens, n_tokens] input, causal 0/-INF

    ggml_tensor * zero_pad = nullptr; // scalar f32 param leaf (grads_needed coverage hack, value 0)
    ggml_tensor * perturb  = nullptr; // f32 [n_embd, n_tokens] additive input at block perturb_layer output

    std::vector<ggml_tensor *> l_out;   // per-block residual outputs [n_embd, n_tokens]
    std::vector<ggml_tensor *> k_out;   // per-block keys after rope [n_embd_head, n_head_kv, n_tokens] (set_output to keep)
    std::vector<ggml_tensor *> v_out;   // per-block values, same shape
    ggml_tensor *              logits = nullptr; // [n_vocab, n_tokens]
};

// build the GLM-4 dense forward graph; n_tokens must be fixed at build time.
// if with_grad_flags: the final block output is flagged GGML_TENSOR_FLAG_LOSS
// (VJP seed point; set directly because ggml_set_loss asserts a scalar shape)
// and the graph/context are created with grads=true and sized for
// ggml_build_backward_expand. The caller must additionally flag one leaf with
// ggml_set_param (ggml requires at least one PARAM) and pass a grad_accs entry
// for the loss node to seed the VJP.
// if perturb_layer >= 0 (requires with_grad_flags): an additive f32 input is
// inserted right after that block's output, for finite-difference checks.
// use_flash_attn: forward-only variant with ggml_flash_attn_ext (small compute
// buffer for long contexts; incompatible with with_grad_flags)
// with_head: build the full-model head (final norm + lm_head over all
// positions); pass false when only block outputs are needed to skip the
// [n_vocab, n_tokens] logits tensor and its GEMM
bool jlens_build_forward(const jlens_weights & m, int n_tokens, bool with_grad_flags, jlens_forward & out,
                         int perturb_layer = -1, bool use_flash_attn = false, bool with_head = true);

// R-lens (LRP backward): two-phase variant of the above.
//
// jlens_build_forward_dt builds the ordinary forward but additionally computes
// the LRP detach values IN-GRAPH and marks them as outputs, so the phase-2
// graph's leaf inputs can alias them directly (no separate leaf buffers):
//   jlens_dt_<il>_rmsinv_<k>  [1, n_tokens]  k = 0 attn, 1 post_attn, 2 ffn, 3 post_ffw
//   jlens_dt_<il>_sig         [n_ff, n_tokens]  sigmoid(gate)
//   jlens_dt_<il>_silu        [n_ff, n_tokens]  silu(gate)
//   jlens_dt_<il>_up          [n_ff, n_tokens]
// dt_one/dt_eps: [1] f32 constant leaves (1.0, norm_eps) owned by the caller.
//
// jlens_build_forward_lrp builds the same network with the LRP rules baked in as
// detached-leaf structure, so ordinary autodiff over it yields the LRP backward:
//   rms_norm(x)*w  -> (x * rms_inv_leaf) * w                 (LN-rule)
//   swiglu(g, u)   -> 0.5*(g*sig_leaf)*up_detached + 0.5*silu_leaf*u
//                                                        (identity + half rule)
// The forward value is identical. All other ops keep their ordinary backward.
struct jlens_lrp_leaves {
    ggml_tensor * rms_inv[4] = {}; // [1, n_tokens], one per norm site
    ggml_tensor * sig_gate   = nullptr; // [n_ff, n_tokens]
    ggml_tensor * silu_gate  = nullptr; // [n_ff, n_tokens]
    ggml_tensor * up         = nullptr; // [n_ff, n_tokens]
};

struct jlens_lrp_forward : jlens_forward {
    std::vector<jlens_lrp_leaves> leaves; // [n_layer], graph inputs to be wired by caller
};

bool jlens_build_forward_dt (const jlens_weights & m, int n_tokens, jlens_forward & out,
                             ggml_tensor * dt_one, ggml_tensor * dt_eps);
bool jlens_build_forward_lrp(const jlens_weights & m, int n_tokens, jlens_lrp_forward & out);

// steering-vector optimization graph: forward with a learnable vector v [n_embd] added to every
// position after block steer_layer, full head, and a soft-label cross-entropy loss against
// labels [n_vocab, n_tokens] (one distribution per row). v is the only PARAM, so the backward
// pass covers blocks > steer_layer and the head only. With per_position, v is [n_embd, n_tokens]
// (one slot per position), so its gradient is the per-position gradient of the loss at that layer.
struct jlens_steer_forward : jlens_forward {
    ggml_tensor * steer  = nullptr; // f32 [n_embd] (or [n_embd, n_tokens] per position) param
    ggml_tensor * labels = nullptr; // f32 [n_vocab, n_tokens] input
    ggml_tensor * loss   = nullptr; // f32 [1], loss-flagged; caller seeds its grad with 1
};
bool jlens_build_forward_steer(const jlens_weights & m, int n_tokens, int steer_layer, jlens_steer_forward & out,
                               bool per_position = false);

// chunk mode of one block: queries at chunk positions [a, a + n_tokens), keys/values for [0, n_kv).
//   main mode: keys/values = k_kv/v_kv with the chunk's own (live) keys/values set at [a, a + n_tokens)
//   now mode:  keys/values = k_kv/v_kv (frozen), plus diagonal terms so that each query's own key/value is live
struct jlens_chunk_kv {
    bool          now    = false;
    int64_t       a      = 0;
    ggml_tensor * k_kv   = nullptr; // [n_embd_head, n_head_kv, n_kv] leaf
    ggml_tensor * v_kv   = nullptr;
    ggml_tensor * one    = nullptr; // main mode: f32 [1] leaf holding 1.0
    ggml_tensor * k_live = nullptr; // out: chunk keys after rope [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor * v_live = nullptr; // out: chunk values
};

// exact value lens, one chunk of positions [a, a + n_tokens) of a text of n_kv tokens (n_kv = a + n_tokens):
// blocks above slot_layer, starting from the slot output h, with the keys/values of the earlier positions as
// leaves, and the loss = sum of next-token NLL at the first n_rows chunk positions.
//   main mode: k_kv/v_kv are PARAMs (their grads go to the earlier chunks), and the loss also has
//              <dk_in, k_chunk> + <dv_in, v_chunk>, the gradient the later chunks sent to this chunk.
//              Then d loss / d h = d (sum of all later NLL, own position included) / d h, swept from the end.
//   now mode:  k_kv/v_kv frozen; d loss / d h = d NLL_s / d h_s (the next-token term) at every position.
// The caller wires the leaves (h, one, k_kv, v_kv, dk_in, dv_in) to device data and fills pos, mask, labels.
struct jlens_upper_chunk {
    ggml_context * gctx = nullptr;
    ggml_cgraph *  gf   = nullptr;
    ggml_tensor *  h      = nullptr; // f32 [n_embd, n_tokens] PARAM leaf
    ggml_tensor *  pos    = nullptr; // i32 [n_tokens] input
    ggml_tensor *  mask   = nullptr; // f32 [n_kv, n_tokens] input
    ggml_tensor *  labels = nullptr; // f32 [n_vocab, n_rows] input, one-hot next tokens (head mode)
    ggml_tensor *  g_top  = nullptr; // f32 [n_embd, n_tokens] leaf: loss gradient at the top_layer output (top_layer < last)
    ggml_tensor *  one    = nullptr; // f32 [1] leaf, caller wires it to a device 1.0
    std::vector<ggml_tensor *> k_kv, v_kv;   // per block above slot_layer
    std::vector<ggml_tensor *> dk_in, dv_in; // main mode only, [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor *  loss   = nullptr; // f32 [1], loss-flagged
};
// top_layer < n_layer - 1 (main mode only): run blocks slot_layer+1 .. top_layer, and the loss is <g_top, h_top> plus the
// key/value terms, so a sweep per layer group chains the exact gradient down the stack. slot_layer = -1: start at the
// token embeddings (h = block-0 input).
bool jlens_build_upper_chunk(const jlens_weights & m, int slot_layer, int64_t a, int n_tokens, int64_t n_kv, int n_rows,
                             bool now_mode, jlens_upper_chunk & out, int top_layer = -1);

// forward-only chunk of positions [a, a + n_tokens) through all blocks, explicit f32 attention over the
// keys/values of [0, n_kv) (k_kv/v_kv leaves hold the earlier positions; the chunk's own are set in).
// The caller wires the leaves and adds the copy-out of k_live/v_live/l_out it needs.
struct jlens_chunk_forward {
    ggml_context * gctx = nullptr;
    ggml_cgraph *  gf   = nullptr;
    ggml_tensor *  tokens = nullptr; // i32 [n_tokens]
    ggml_tensor *  pos    = nullptr; // i32 [n_tokens]
    ggml_tensor *  mask   = nullptr; // f32 [n_kv, n_tokens]
    ggml_tensor *  one    = nullptr; // f32 [1] leaf holding 1.0
    ggml_tensor *  emb    = nullptr; // [n_embd, n_tokens] block-0 input (token embeddings, + emb_add)
    ggml_tensor *  emb_add = nullptr; // with_emb_add: f32 [n_embd, n_tokens] leaf added to the embeddings
    std::vector<ggml_tensor *> k_kv, v_kv;     // per block, [n_embd_head, n_head_kv, n_kv] leaves
    std::vector<ggml_tensor *> k_live, v_live; // per block, chunk keys (after rope) / values
    std::vector<ggml_tensor *> l_out;          // per block output [n_embd, n_tokens]
};
bool jlens_build_chunk_forward(const jlens_weights & m, int64_t a, int n_tokens, int64_t n_kv, jlens_chunk_forward & out,
                               bool with_emb_add = false);

// exact value lens for a whole text (jlens-value.cpp). For each position s, with h_s the slot_layer output:
//   now  = d log p(x_{s+1}) / d h_s
//   fut  = sum over all later t > s of d log p(x_{t+1}) / d h_s
// A forward sweep over chunks (explicit f32 attention; flash attention's f16 keys/values change these gradients
// by several %), then a backward sweep from the end (keys/values of earlier chunks frozen, their gradient
// carried back), then one pass of now-mode chunks. full_fwd: one unchunked forward instead (O(N^2) memory,
// a reference for checks).
struct jlens_value_result {
    std::vector<float> fut_norm, now_norm; // [n_tokens]
    std::vector<float> fut, now;           // [n_tokens][n_embd] if keep_vectors
    std::vector<float> h;                  // [n_tokens][n_embd] slot output, if keep_vectors
    int    chunk = 0;
    double sec_fwd = 0, sec_main = 0, sec_now = 0;
};
bool jlens_value_exact(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                       const std::vector<int32_t> & tokens, int slot_layer, int chunk, bool keep_vectors,
                       jlens_value_result & out, std::string & err, bool full_fwd = false);

// branch gradients (jlens-value.cpp). For each position s and each of the model's top-k candidates c for x_{s+1}:
//   delta = linearized change in sum over t > s of log p(x_{t+1}) if x_{s+1} were c instead of the actual token
//         = g . (E[c] - E[x_{s+1}]),  g = d (that sum) / d (token embedding at s+1)
// and spread = sqrt(sum p (delta - mean)^2) over the top-k (p renormalized): how much the rest of the text depends
// on which candidate is chosen. g comes from a chunked backward through all blocks, in layer groups of <= 21
// blocks (the gradient at each group's bottom is fed to the next group down). n_real: number of real vocab rows
// in the output (0 = all).
struct jlens_branch_result {
    int k = 0, chunk = 0;
    std::vector<int32_t> cand;   // [n_tokens][k] candidate token ids for x_{s+1}
    std::vector<float>   prob;   // [n_tokens][k]
    std::vector<float>   delta;  // [n_tokens][k], NaN where undefined (s >= n_tokens - 2)
    std::vector<float>   spread; // [n_tokens], NaN where undefined
    double sec_fwd = 0, sec_bwd = 0;
};
bool jlens_branch_grad(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                       const std::vector<int32_t> & tokens, int topk, int chunk, int n_real, jlens_branch_result & out,
                       std::string & err, std::vector<float> * grad_emb = nullptr, int max_group = 21);

// for checks: per-position log p(x_{s+1}) from the chunked forward, with emb_add (host [n_tokens][n_embd], or null)
// added to the token embeddings; and dequantized token embedding rows
bool jlens_text_loglik(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                       const std::vector<int32_t> & tokens, int chunk, int n_real, const float * emb_add,
                       std::vector<double> & ll, std::string & err);
bool jlens_embed_rows(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                      const std::vector<int32_t> & ids, std::vector<float> & rows, std::string & err);

// single-block graph for local validation: h_in leaf -> (+zero_pad param) -> block -> h_out (loss)
struct jlens_block {
    ggml_context * gctx = nullptr;
    ggml_cgraph *  gf = nullptr;
    ggml_tensor *  h_in = nullptr;     // f32 [n_embd, n_tokens] input leaf
    ggml_tensor *  x = nullptr;        // h_in + zero_pad: backward grad capture point
    ggml_tensor *  zero_pad = nullptr; // scalar f32 param (value 0)
    ggml_tensor *  pos = nullptr;
    ggml_tensor *  mask = nullptr;
    ggml_tensor *  h_out = nullptr;    // loss-flagged [n_embd, n_tokens]
};
bool jlens_build_block(const jlens_weights & m, int il, int n_tokens, jlens_block & out);

// build graph: mul_mat(m.get("output.weight"), rms_norm(x)*output_norm) -> [n_vocab, 1]
// used for logit-lens readout of an arbitrary activation column
ggml_cgraph * jlens_build_head(const jlens_weights & m, ggml_context * gctx, ggml_tensor ** x_in);
