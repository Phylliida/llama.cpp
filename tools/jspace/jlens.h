#pragma once

// J-lens tooling: load a GLM-4 dense GGUF into a ggml CUDA buffer and build
// custom forward graphs that expose every block's residual-stream output.
//
// The graphs are built by hand (not via llama.cpp's graph builders) so that
//   - every op has a working ggml backward path (no flash attention, split swiglu)
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

struct jlens_model {
    jlens_hparams hparams;

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

    ggml_tensor * get(const std::string & name) const;
    ggml_tensor * blk(const char * fmt, int il) const; // fmt like "blk.%d.attn_q.weight"
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
    ggml_tensor *              logits = nullptr; // [n_vocab, n_tokens]
};

// build the GLM-4 dense forward graph; n_tokens must be fixed at build time.
// if with_grad_flags: the final block output gets ggml_set_loss (VJP seed point)
// and the graph/context are created with grads=true and sized for
// ggml_build_backward_expand. The caller must additionally flag one leaf with
// ggml_set_param (ggml requires at least one PARAM) and pass a grad_accs entry
// for the loss node to seed the VJP.
// if perturb_layer >= 0 (requires with_grad_flags): an additive f32 input is
// inserted right after that block's output, for finite-difference checks.
bool jlens_build_forward(const jlens_model & m, int n_tokens, bool with_grad_flags, jlens_forward & out,
                         int perturb_layer = -1);

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
bool jlens_build_block(const jlens_model & m, int il, int n_tokens, jlens_block & out);

// build graph: mul_mat(m.get("output.weight"), rms_norm(x)*output_norm) -> [n_vocab, 1]
// used for logit-lens readout of an arbitrary activation column
ggml_cgraph * jlens_build_head(const jlens_model & m, ggml_context * gctx, ggml_tensor ** x_in);
