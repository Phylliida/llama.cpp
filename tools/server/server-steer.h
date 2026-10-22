#pragma once

// /steer: steered and unsteered continuations from a point in a text, along one direction probe (--probe-dirs).
//
// The push mirrors homeo-ident (plan B in base-model-welfare's NOTES-homeostasis.md): dose x mean |h_L| along the unit
// direction u, added after block L from the push token on ("step", the input kappa is measured with) or at the push
// token only ("impulse"). The text before the push token is decoded once, unsteered, and shared by every arm through
// the KV cache. Common random numbers: one uniform per (sequence, step), shared by all arms, so the arms differ only
// through the push. At every decoded position the PRE-push layer-L state along u is recorded: a later position sees
// the push at layer L only through the tokens written, so its change against the unsteered arm is the closed-loop,
// token-mediated response. Humane endpoints as homeo-ident: a sequence stops on a loop or after 16 steps with
// next-token entropy > 6 nats. Every request with a push appends a row to the exposure log (--steer-log).
//
// Runs in its own llama_context (created on the first request; 8192 cells of unified KV), so it never touches the
// server's slots. Requests are serialized.

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

struct llama_model;
struct llama_vocab;

// after every model (re)load; log_path empty = exposure rows go to the server log only
void server_steer_init(llama_model * model, const std::string & model_name, const std::string & log_path, float max_dose);
// before the model is freed (sleep/shutdown); blocks until a running request finishes; writes a pending group row
void server_steer_deinit();

struct server_steer_params {
    std::vector<int32_t> prefix;          // tokens up to and including the push token
    std::vector<int32_t> follow;          // teacher-forced: the existing text after the push token (may be empty)
    std::string          dir;             // direction probe name, or empty with vector + layer
    std::vector<float>   vector;          // a raw direction (normalized here), e.g. a random control
    int                  layer = -1;      // its layer
    bool                 set = false;     // impulse only: one pushed arm that sets u.h / |h| at the push token to set_cos
    float                set_cos = 0.0f;
    std::vector<std::pair<std::string, std::vector<int32_t>>> classes; // teacher-forced: log P(class) per position
    std::vector<float>   doses = {0.25f, -0.25f}; // in units of mean |h_L|; an unsteered arm always runs first
    bool                 impulse = false; // push at the push token only (default: from it on)
    int                  n_seq   = 4;     // sequences per arm (common random numbers across arms)
    int                  n_gen   = 48;    // tokens sampled per sequence
    uint64_t             seed    = 1;
    float                temp    = 1.0f;
    int                  n_top   = 8;     // teacher-forced: tokens listed per side at the push token
    std::string          note;            // goes into the exposure log
    std::string          log_group;       // non-empty: one exposure row per group (written when the group changes, every
                                          // 100 requests or 10 minutes, on log_flush and at shutdown), not per request
    bool                 log_flush = false;
    bool                 baseline = true;  // run the unsteered arm (arm 0); false: skip it (streaming, the loom's generation)
    // called with each sampled token (arm, sequence, token), in order per sequence; return false to stop generating
    // (the run then ends normally with what it has, and is logged)
    std::function<bool(int, int, int32_t)> on_token;
};

struct server_steer_seq {
    std::vector<int32_t> tok;        // sampled tokens (the last one is the endpoint's, if any)
    std::vector<float>   proj;       // per decoded step (step 0 = the push token): pre-push u.h
    std::vector<float>   norm;       // |pre-push h|
    std::vector<float>   ent;        // entropy of the sampling distribution (nats)
    int                  first_div = -1; // first step whose token differs from the unsteered arm's (-1: none)
    int                  endpoint  = 0;  // 0 none, 1 loop, 2 incoherent
};

// teacher-forced effect of one arm on the existing text: position k = push token + k predicts follow[k]
struct server_steer_tf {
    std::vector<float> kl;     // [n_follow + 1] KL(unsteered || this arm) of the next-token distribution (nats)
    std::vector<float> dlogp;  // [n_follow] change in log p of the actual next token
    std::vector<std::pair<int32_t, float>> up, down; // at the push token: tokens whose log p rises / falls most
    std::vector<std::vector<float>> class_logp; // [n_class][n_follow + 1] log P(class) of the next token
};

struct server_steer_arm {
    float                         dose = 0.0f;
    float                         push = 0.0f; // dose * h_norm
    std::vector<server_steer_seq> seqs;
    server_steer_tf               tf;
};

struct server_steer_result {
    int                           layer  = -1;
    float                         h_norm = 0.0f;
    float                         proj0  = 0.0f; // set mode: u.h and |h| at the push token before the push
    float                         norm0  = 0.0f;
    std::vector<server_steer_arm> arms; // arms[0] unsteered
    double                        sec = 0.0;
};

bool server_steer_run(const llama_vocab * vocab, const server_steer_params & p, server_steer_result & out, std::string & err);
