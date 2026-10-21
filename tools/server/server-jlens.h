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

// load transport matrices + initialize the compute backend (on the CUDA device
// holding the model weights). Called after every model (re)load, including
// wake-from-sleep; idempotent (deinitializes any previous state first).
// Returns false on error (err set).
bool server_jlens_init(const llama_model * model, const std::string & jlens_path, std::string & err);

// release all jlens state. Must be called before the model is destroyed
// (sleep/shutdown): the weights adapter holds raw ggml_tensor pointers into
// the llama model. Blocks until any in-flight compute finishes.
void server_jlens_deinit();

bool server_jlens_enabled();

// run the readout. layers == nullptr => default (every 5th layer + last).
// returns false on error (err set). Token pieces are produced via the vocab.
struct server_jlens_layer_result {
    int layer;
    std::vector<std::pair<std::string, float>> topk; // (token piece, logit), single-position mode
    // all-positions mode: topk per position (parallel to the token array)
    std::vector<std::vector<std::pair<std::string, float>>> positions;
};

// linear readouts on one block output (value-lens probes from
// homeo/value_readout_export.py): PATH.f32 holds [n][n_embd] weights then [n]
// biases, PATH.json {"layer", "n_embd", "names"}. readout = W h + b per position.
// Load after server_jlens_init; survives model reloads.
bool server_jlens_load_readouts(const std::string & path, std::string & err);
bool server_jlens_readouts_enabled();

struct server_jlens_readouts {
    int layer = -1;
    std::vector<std::string> names;
    std::vector<std::vector<float>> values; // [n_tokens][n_names]
};

// direction probes at several layers (layer-sweep directions from homeo/score_probe_export.py in
// base-model-welfare): PATH.f32 holds [n][n_embd] directions, PATH.json {"n_embd", "names", "layers"} (one layer
// per direction). Per position, u.h and u.h / |h| at the direction's layer. Load after server_jlens_init; survives
// model reloads.
bool server_jlens_load_dirs(const std::string & path, std::string & err);
bool server_jlens_dirs_enabled();
// one direction by name: its layer, unit vector and the mean |h| at its layer in natural text (PATH.json "h_norms";
// 0 if absent). false if unknown.
bool server_jlens_dir_get(const std::string & name, int & layer, float & h_norm, std::vector<float> & u);
// mean |h| at a layer from any probe direction there (0 if none)
float server_jlens_layer_h_norm(int layer);

struct server_jlens_dirs {
    std::vector<std::string> names;          // the selected directions
    std::vector<int> layers;                 // their layers
    std::vector<std::vector<float>> values;  // [n_tokens][n_sel] u.h
    std::vector<std::vector<float>> cos;     // [n_tokens][n_sel] u.h / |h|
};

// linear probes on all block outputs at once (all-layer probes from homeo/lin_probes_export.py in base-model-welfare):
// PATH.f32 holds [n][n_layer][n_embd] weights then [n] biases (raw block outputs, value = sum_L w_L . h_L + b),
// PATH.json {"n_embd", "n_layer", "names"}. Computed inside the forward graph. Load after server_jlens_init; survives
// model reloads.
bool server_jlens_load_lin(const std::string & path, std::string & err);
bool server_jlens_lin_enabled();

struct server_jlens_lin {
    std::vector<std::string> names;
    std::vector<std::vector<float>> values; // [n_tokens][n]
};

// per-position next-token statistics from the model's own output distribution
struct server_jlens_token_stats {
    std::vector<float> surprisal; // [n_tokens] -log p(x_{s+1}); NaN at the last position
    std::vector<float> entropy;   // [n_tokens] entropy of the next-token distribution (nats)
    std::vector<float> branches;  // [n_tokens] 1 / sum p^2: effective number of strong candidates
    std::vector<float> runner_up; // [n_tokens] p2 / p1: second candidate relative to the first
};

// readouts != nullptr: also compute the readouts at every position.
// stats != nullptr: also compute surprisal and entropy at every position.
// lin != nullptr: also the all-layer linear probes at every position.
// dirs != nullptr: also project on the direction probes, those at dir_layers only (all when null or empty). Each
// selected layer's block output stays in VRAM until the end of the pass (n_tokens * n_embd floats per layer).
// readouts_only: skip the J-lens decode (layers and model_topk stay empty).
bool server_jlens_compute(
        const llama_vocab * vocab,
        const std::vector<int32_t> & tokens,
        int pos,
        int topk,
        const std::vector<int> * layers,
        bool all_positions,
        std::vector<server_jlens_layer_result> & out_layers,
        std::vector<std::pair<std::string, float>> & out_model_topk,
        std::string & err,
        server_jlens_readouts * readouts = nullptr,
        bool readouts_only = false,
        server_jlens_token_stats * stats = nullptr,
        server_jlens_dirs * dirs = nullptr,
        const std::vector<int> * dir_layers = nullptr,
        server_jlens_lin * lin = nullptr);

// exact value lens (jlens_value_exact) at slot_layer for every position: |d log p(x_{s+1}) / d h_s| (now) and
// |sum over all later t > s of d log p(x_{t+1}) / d h_s| (fut). Needs only the model weights (no probe file).
bool server_jlens_value_exact(const std::vector<int32_t> & tokens, int slot_layer, int chunk,
                              std::vector<float> & fut_norm, std::vector<float> & now_norm, double & sec, std::string & err);

// branch gradients (jlens_branch_grad): per position the spread of the linearized effect of the top-k candidates
// for the next token on the log-likelihood of all later tokens, and the candidates themselves
struct server_jlens_branch_cand {
    std::string piece;
    float       prob;
    float       delta; // linearized change in the later log-likelihood if this were the next token (NaN if undefined)
};
struct server_jlens_branch {
    int k = 0;
    std::vector<float> spread;                                  // [n_tokens], NaN where undefined
    std::vector<std::vector<server_jlens_branch_cand>> cand;    // [n_tokens][k]
    double sec = 0.0;
};
bool server_jlens_branch_grad(const llama_vocab * vocab, const std::vector<int32_t> & tokens, int topk, int chunk,
                              server_jlens_branch & out, std::string & err);
