#pragma once

// /jlens endpoint: apply fitted J-lens transport matrices to per-layer residual
// outputs of a prompt, decode through the model's final norm + unembedding, and
// return per-layer top-k tokens. Uses the model weights already loaded by
// llama-server (no second copy of the model) with a hand-built ggml graph.
//
// The JLNS file is the raw accumulator format written by jlens-fit; the
// transport matrices are normalized to J = M / n_positions at load time.

#include <string>
#include <vector>

struct llama_model;
struct llama_vocab;

// load transport matrices + initialize the compute backend (CUDA device 0).
// call once after the model is loaded. Returns false on error (err set).
bool server_jlens_init(const llama_model * model, const std::string & jlens_path, std::string & err);

bool server_jlens_enabled();

// run the readout. layers == nullptr => default (every 5th layer + last).
// returns false on error (err set). Token pieces are produced via the vocab.
struct server_jlens_layer_result {
    int layer;
    std::vector<std::pair<std::string, float>> topk; // (token piece, logit)
};

bool server_jlens_compute(
        const llama_vocab * vocab,
        const std::vector<int32_t> & tokens,
        int pos,
        int topk,
        const std::vector<int> * layers,
        std::vector<server_jlens_layer_result> & out_layers,
        std::vector<std::pair<std::string, float>> & out_model_topk,
        std::string & err);
