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

#include <cmath>
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

// guided sampling (/steer with "guide"): no push. At every step the most likely candidates for the next token (top_p of
// the probability, at most k) are decoded as one batch, each in its own sequence on the text so far, and scored by a
// metric of what follows each one. The token is the best candidate (maximize) or is sampled from p(x) exp(beta z(x))
// over the candidates, where z is the metric on the loom's color scale (probit of its percentile). Same context, humane
// endpoints and exposure log as /steer.
struct server_guide_params {
    std::vector<int32_t> prefix;
    // lin, lind, lintd: all-layer probe f, its change f(x) - f(prev), TD error -c(x) - horizon (f(x) - f(prev)) (--lin-probes)
    // readout: value-lens readout (--value-probe); dir: direction probe u.h / |h| (--probe-dirs)
    // entropy, branches, runner_up: of the next-token distribution after the candidate
    // surprisal, gap: -log p(x) and -log p(x) - entropy, of the candidate itself
    std::string        kind;
    std::string        name;            // the probe (lin*, readout, dir)
    float              horizon = 0.0f;  // lintd
    std::vector<float> pct_q, pct_v;    // color scale: value pct_v[i] at percentile pct_q[i] (ascending)
    float              sign     = 1.0f; // -1: lower is better (the loom's flip)
    bool               maximize = false;
    float              beta     = 1.0f;
    int                k        = 32;
    float              top_p    = 0.95f;
    int                n_gen    = 128;
    uint64_t           seed     = 1;
    float              temp     = 1.0f;
    std::string        note;
    std::string        log_group;
    bool               log_flush = false;
    // called with each chosen token, its metric value and z; return false to stop
    std::function<bool(int32_t, float, float)> on_token;
};

struct server_guide_result {
    std::vector<int32_t> tok;
    std::vector<float>   value;   // per step: the chosen token's metric value
    std::vector<float>   z;       // its z (color scale, before sign)
    std::vector<float>   z_model; // mean z of the candidates under the model's own probabilities (what plain sampling would give)
    std::vector<float>   ent;     // entropy of the sampling distribution (nats)
    std::vector<int>     n_cand;
    int                  endpoint = 0; // 0 none, 1 loop, 2 incoherent
    double               sec = 0.0;
};

bool server_guide_run(const llama_vocab * vocab, const server_guide_params & p, server_guide_result & out, std::string & err);

// superposed rollout (/steer with "superpose"): no sampling. At every step the next input is the probability-weighted
// mean of the most likely tokens' input embeddings (the top k, cut at top_p of the probability, renormalized over them),
// fed as a raw embedding, so the text runs forward in superposition; the loom shows the most likely token as the text.
// The top mix_k candidates are also decoded on their own, each in its own sequence on the same superposed past (one batch
// with the superposed input), which gives the mixture of their next-token distributions: how far the superposed input's
// next-token distribution is from that mixture is the interference at that step. Same context, humane endpoints and exposure log
// as /steer.
struct server_superpose_step {
    int32_t              tok    = -1;   // the most likely candidate
    std::vector<int32_t> top;           // the candidates, most likely first (at most n_show)
    std::vector<float>   top_w;         // their weights in the input (renormalized over all the candidates)
    float                ent    = 0.0f; // entropy of the next-token distribution (nats, at temp)
    float                eff    = 0.0f; // 1 / sum w^2 over the candidates: the effective number of tokens in the input
    float                mass   = 0.0f; // probability of the candidates before renormalizing
    int                  n_cand = 0;
    float                shrink = 0.0f; // |input| / sum w |e(x)| (before norm): how much averaging shortened it
    // from the next-token distributions after this input: q after the superposed input, p_x after candidate x alone,
    // m = sum w p_x the mixture that a superposition without interference would give (NAN without mix)
    float                kl      = NAN; // KL(m || q), nats
    float                tv      = NAN; // TV(m, q)
    float                tv_top  = NAN; // TV(q, p_top): small when the input acted like its top token alone (collapse)
    int32_t              near    = -1;  // the candidate whose own p_x is closest to q (TV)
    float                near_w  = NAN; // its weight in the input
    float                tv_near = NAN;
    float                mix_mass = NAN; // probability of the decoded candidates (the mixture is renormalized over them)
};

struct server_superpose_params {
    std::vector<int32_t> prefix;
    int         k      = 512;
    float       top_p  = 1.0f;
    float       temp   = 1.0f;
    int         n_gen  = 128;
    int         n_show = 8;
    bool        norm   = false; // rescale the input to sum w |e(x)| (undo the shrink)
    bool        mix    = true;  // decode the top mix_k candidates too (interference metrics; mix_k more rows per step)
    int         mix_k  = 32;
    std::string note;
    std::string log_group;
    bool        log_flush = false;
    // called after every step; return false to stop
    std::function<bool(const server_superpose_step &)> on_step;
};

struct server_superpose_result {
    std::vector<server_superpose_step> steps;
    int    endpoint = 0; // 0 none, 1 loop, 2 incoherent
    double sec      = 0.0;
};

bool server_superpose_run(const llama_vocab * vocab, const server_superpose_params & p, server_superpose_result & out, std::string & err);
