// homeo-band: paired rollouts for probability-band steering vectors (S2b in NOTES-homeostasis.md).
//
// At prefix ends taken from each rollout (--prefix-lens), the KV cache is branched into arms
// that continue for --n-steps tokens. All arms are decoded in one batch per step (same kernels):
//   arm 0          normal: sample from p (temp 1, no truncation)
//   arm 1..L       band:P  sample prop. to p among tokens with p in [P/w, P*w]; if none, fall back to
//                  normal sampling (--fallback normal) or the token with log p nearest log P (nearest)
//   arm L+1..2L    temp:P  (only with --temp-arms) sample from p^(1/T), T set per step so that E[-ln p] = -ln P
// l_out-N is captured for the last --n-capture steps of every arm.
//
// usage: homeo-band -m model.gguf --rollouts rollouts.jsonl --out DIR [--levels 0.5,0.25,0.1,0.03]
//          [--width 1.5] [--prefix-lens 256,1024] [--n-steps 40] [--n-capture 8]
//          [--layers 16,24,32,40,44,48] [--n-rollouts N] [--seed 1] [--fallback normal|nearest] [--temp-arms]
//
// outputs in DIR:
//   steps.bin     step_rec per (prefix, arm, step)
//   samples.bin   sample_rec per captured row, acts_LNN.f16 rows match
//   meta.json

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using json = nlohmann::ordered_json;

static const int MAX_LEVELS = 8;

struct step_rec {
    int32_t prefix;
    int32_t arm;
    int32_t step;
    int32_t token;
    float   p;        // p of the sampled token under the arm's own current distribution
    float   entropy;  // H(p) before sampling
    float   T;        // temperature used (1 for normal/band)
    int32_t band_hit; // band arm: candidate set was non-empty
    float   band_mass[MAX_LEVELS]; // mass of p inside each level's band
};
static_assert(sizeof(step_rec) == 64, "step_rec layout");

struct sample_rec {
    int32_t prefix;
    int32_t arm;
    int32_t step;
    int32_t token;
    int32_t seq;     // source rollout
    int32_t pos;     // absolute position
    float   p;
    float   entropy;
};
static_assert(sizeof(sample_rec) == 32, "sample_rec layout");

static void die(const char * msg) {
    fprintf(stderr, "homeo-band: %s\n", msg);
    exit(1);
}

template <class F>
static void parallel_for(int n, int n_threads, const F & f) {
    const int chunk = (n + n_threads - 1) / n_threads;
    std::vector<std::thread> th;
    for (int t = 0; t < n_threads; ++t) {
        const int a = t * chunk, b = std::min(n, a + chunk);
        if (a >= b) {
            break;
        }
        th.emplace_back([a, b, &f] { for (int i = a; i < b; ++i) f(i); });
    }
    for (auto & x : th) {
        x.join();
    }
}

static std::vector<float> parse_floats(const std::string & s) {
    std::vector<float> out;
    char * c = strdup(s.c_str());
    for (char * t = strtok(c, ","); t; t = strtok(nullptr, ",")) out.push_back((float) atof(t));
    free(c);
    return out;
}

// ---- l_out capture (only while enabled) ----

struct capture {
    std::unordered_map<std::string, int> slot_of;
    std::vector<std::vector<float>>      buf;
    std::vector<int64_t>                 rows;
    int64_t                              n_embd  = 0;
    bool                                 enabled = false;
};

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    auto * c = (capture *) ud;
    if (!c->enabled) {
        return !ask;
    }
    auto it = c->slot_of.find(t->name);
    if (it == c->slot_of.end()) {
        return !ask;
    }
    if (ask) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd || !ggml_is_contiguous(t)) {
        die("unexpected l_out tensor layout");
    }
    const int s = it->second;
    c->rows[s] = ggml_nrows(t);
    c->buf[s].resize(c->rows[s] * c->n_embd);
    ggml_backend_tensor_get(t, c->buf[s].data(), 0, ggml_nbytes(t));
    return true;
}

// ---- samplers on log p ----

static void log_softmax(const float * z, float * out, int n) {
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) m = std::max(m, z[i]);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp(z[i] - m);
    const float lse = m + (float) std::log(s);
    for (int i = 0; i < n; ++i) out[i] = z[i] - lse;
}

static int sample_weighted(const std::vector<double> & w, std::mt19937_64 & rng) {
    double tot = 0.0;
    for (double x : w) tot += x;
    double u = std::uniform_real_distribution<double>(0.0, tot)(rng);
    for (size_t i = 0; i < w.size(); ++i) {
        u -= w[i];
        if (u <= 0.0) return (int) i;
    }
    return (int) w.size() - 1;
}

static double tempered_surprisal(const float * lp, int n, double T) {
    double m = -INFINITY;
    for (int i = 0; i < n; ++i) m = std::max(m, (double) lp[i] / T);
    double s = 0.0, sl = 0.0;
    for (int i = 0; i < n; ++i) {
        const double e = std::exp(lp[i] / T - m);
        s  += e;
        sl += e * lp[i];
    }
    return -sl / s;
}

int main(int argc, char ** argv) {
    std::string model_path, rollouts_path, out_dir;
    std::string levels_str = "0.5,0.25,0.1,0.03", prefix_str = "256,1024", layers_str = "16,24,32,40,44,48";
    float width     = 1.5f;
    int   n_steps   = 40;
    int   n_capture = 8;
    int   n_rollouts = -1;
    int   seed      = 1;
    int   ubatch    = 512;
    int   n_threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    bool  fallback_nearest = false;
    bool  temp_arms = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) die(("missing value for " + a).c_str()); return argv[++i]; };
        if      (a == "-m")            model_path    = next();
        else if (a == "--rollouts")    rollouts_path = next();
        else if (a == "--out")         out_dir       = next();
        else if (a == "--levels")      levels_str    = next();
        else if (a == "--width")       width         = (float) atof(next());
        else if (a == "--prefix-lens") prefix_str    = next();
        else if (a == "--n-steps")     n_steps       = atoi(next());
        else if (a == "--n-capture")   n_capture     = atoi(next());
        else if (a == "--layers")      layers_str    = next();
        else if (a == "--n-rollouts")  n_rollouts    = atoi(next());
        else if (a == "--seed")        seed          = atoi(next());
        else if (a == "--threads")     n_threads     = atoi(next());
        else if (a == "--fallback")    fallback_nearest = std::string(next()) == "nearest";
        else if (a == "--temp-arms")   temp_arms     = true;
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || rollouts_path.empty() || out_dir.empty()) {
        fprintf(stderr, "usage: homeo-band -m model.gguf --rollouts rollouts.jsonl --out DIR [options]\n");
        return 1;
    }
    const std::vector<float> levels = parse_floats(levels_str);
    std::vector<int> prefix_lens, layers;
    for (float x : parse_floats(prefix_str)) prefix_lens.push_back((int) x);
    for (float x : parse_floats(layers_str)) layers.push_back((int) x);
    std::sort(prefix_lens.begin(), prefix_lens.end());
    const int L = (int) levels.size();
    const int K = 1 + (temp_arms ? 2 : 1) * L;
    if (L < 1 || L > MAX_LEVELS) die("need 1..8 levels");
    if (n_capture > n_steps) die("--n-capture > --n-steps");
    const float lw = std::log(width);
    std::vector<float> llev;
    for (float x : levels) llev.push_back(std::log(x));

    std::vector<std::pair<std::vector<int32_t>, std::string>> rollouts;
    {
        std::ifstream f(rollouts_path);
        if (!f) die("cannot open rollouts file");
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const json j = json::parse(line);
            rollouts.push_back({ j.at("prompt_ids").get<std::vector<int32_t>>(), j.at("text").get<std::string>() });
        }
    }
    if (n_rollouts >= 0 && (size_t) n_rollouts < rollouts.size()) rollouts.resize(n_rollouts);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) die("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);
    for (int il : layers) if (il < 0 || il >= llama_model_n_layer(model)) die("--layers out of range");

    capture cap;
    cap.n_embd = n_embd;
    cap.buf.resize(layers.size());
    cap.rows.resize(layers.size());
    for (size_t s = 0; s < layers.size(); ++s) cap.slot_of["l_out-" + std::to_string(layers[s])] = (int) s;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = prefix_lens.back() + K * n_steps + 256;
    cp.n_batch           = ubatch;
    cp.n_ubatch          = ubatch;
    cp.n_seq_max         = K + 1;
    cp.kv_unified        = true;
    cp.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval           = cb_eval;
    cp.cb_eval_user_data = &cap;
    cp.n_threads         = n_threads;
    cp.n_threads_batch   = n_threads;
    cp.no_perf           = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context init failed");
    llama_memory_t mem = llama_get_memory(ctx);

    if (system(("mkdir -p '" + out_dir + "'").c_str()) != 0) die("mkdir failed");
    FILE * f_steps   = fopen((out_dir + "/steps.bin").c_str(), "wb");
    FILE * f_samples = fopen((out_dir + "/samples.bin").c_str(), "wb");
    if (!f_steps || !f_samples) die("cannot open output files");
    std::vector<FILE *> f_acts;
    for (int il : layers) {
        char name[64];
        snprintf(name, sizeof(name), "/acts_L%02d.f16", il);
        FILE * f = fopen((out_dir + name).c_str(), "wb");
        if (!f) die("cannot open acts file");
        f_acts.push_back(f);
    }

    llama_batch batch = llama_batch_init(std::max(ubatch, K), 0, 1);
    std::vector<std::vector<float>> lp(K, std::vector<float>(n_vocab));
    std::vector<float> zprev(n_vocab);
    std::vector<ggml_fp16_t> row16(n_embd);
    int64_t n_prefix = 0, n_samples = 0;
    const auto t_start = std::chrono::steady_clock::now();

    for (size_t r = 0; r < rollouts.size(); ++r) {
        std::vector<int32_t> toks = rollouts[r].first;
        {
            const std::string & text = rollouts[r].second;
            std::vector<llama_token> tt(text.size() + 16);
            const int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tt.data(), (int32_t) tt.size(), false, true);
            if (n < 0) die("tokenize failed");
            toks.insert(toks.end(), tt.begin(), tt.begin() + n);
        }

        llama_memory_clear(mem, true);
        int cur = 0;
        for (int a : prefix_lens) {
            if (a > (int) toks.size()) break;

            // extend the main sequence to [0, a); only the last row needs logits
            cap.enabled = false;
            while (cur < a) {
                const int n = std::min(ubatch, a - cur);
                batch.n_tokens = n;
                for (int j = 0; j < n; ++j) {
                    batch.token[j] = toks[cur + j]; batch.pos[j] = cur + j; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
                    batch.logits[j] = cur + j == a - 1;
                }
                if (llama_decode(ctx, batch) != 0) die("llama_decode failed (prefix)");
                cur += n;
            }
            std::memcpy(zprev.data(), llama_get_logits_ith(ctx, -1), sizeof(float) * n_vocab);
            for (int k = 0; k < K; ++k) llama_memory_seq_cp(mem, 0, k + 1, 0, a);

            std::vector<std::mt19937_64> rng;
            for (int k = 0; k < K; ++k) rng.emplace_back((uint64_t) seed * 1000003ull + r * 7919ull + a * 104729ull + k);
            std::vector<int32_t> tok(K);
            std::vector<step_rec> srec(K);

            std::vector<const float *> zrow(K);
            for (int s = 0; s < n_steps; ++s) {
                for (int k = 0; k < K; ++k) zrow[k] = s == 0 ? zprev.data() : llama_get_logits_ith(ctx, k);
                parallel_for(K, K, [&](int k) {
                    log_softmax(zrow[k], lp[k].data(), n_vocab);
                    const float * l = lp[k].data();
                    step_rec & st = srec[k];
                    st = {};
                    st.prefix = (int32_t) n_prefix; st.arm = k; st.step = s; st.T = 1.0f;
                    double H = 0.0;
                    for (int v = 0; v < n_vocab; ++v) {
                        const double p = std::exp((double) l[v]);
                        H -= p * l[v];
                        for (int b = 0; b < L; ++b) {
                            if (std::fabs(l[v] - llev[b]) <= lw) st.band_mass[b] += (float) p;
                        }
                    }
                    st.entropy = (float) H;

                    int choice = -1;
                    if (k == 0) {
                        std::vector<double> w(n_vocab);
                        for (int v = 0; v < n_vocab; ++v) w[v] = std::exp((double) l[v]);
                        choice = sample_weighted(w, rng[k]);
                    } else if (k <= L) {
                        const float target = llev[k - 1];
                        std::vector<int> cand;
                        std::vector<double> w;
                        int nearest = 0;
                        for (int v = 0; v < n_vocab; ++v) {
                            if (std::fabs(l[v] - target) <= lw) { cand.push_back(v); w.push_back(std::exp((double) l[v])); }
                            if (std::fabs(l[v] - target) < std::fabs(l[nearest] - target)) nearest = v;
                        }
                        st.band_hit = !cand.empty();
                        if (!cand.empty()) {
                            choice = cand[sample_weighted(w, rng[k])];
                        } else if (fallback_nearest) {
                            choice = nearest;
                        } else {
                            std::vector<double> wn(n_vocab);
                            for (int v = 0; v < n_vocab; ++v) wn[v] = std::exp((double) l[v]);
                            choice = sample_weighted(wn, rng[k]);
                        }
                    } else {
                        const double target = -std::log(levels[k - 1 - L]);
                        double lo = std::log(0.02), hi = std::log(50.0);
                        for (int it = 0; it < 30; ++it) {
                            const double mid = 0.5 * (lo + hi);
                            if (tempered_surprisal(l, n_vocab, std::exp(mid)) < target) lo = mid; else hi = mid;
                        }
                        const double T = std::exp(0.5 * (lo + hi));
                        st.T = (float) T;
                        float m = -INFINITY;
                        for (int v = 0; v < n_vocab; ++v) m = std::max(m, (float) (l[v] / T));
                        std::vector<double> w(n_vocab);
                        for (int v = 0; v < n_vocab; ++v) w[v] = std::exp(l[v] / T - m);
                        choice = sample_weighted(w, rng[k]);
                    }
                    tok[k]   = choice;
                    st.token = choice;
                    st.p     = std::exp(l[choice]);
                });
                fwrite(srec.data(), sizeof(step_rec), K, f_steps);

                cap.enabled = s >= n_steps - n_capture;
                std::fill(cap.rows.begin(), cap.rows.end(), 0);
                batch.n_tokens = K;
                for (int k = 0; k < K; ++k) {
                    batch.token[k] = tok[k]; batch.pos[k] = a + s; batch.n_seq_id[k] = 1; batch.seq_id[k][0] = k + 1;
                    batch.logits[k] = 1;
                }
                if (llama_decode(ctx, batch) != 0) die("llama_decode failed (arms)");
                if (cap.enabled) {
                    for (int64_t rr : cap.rows) if (rr != K) die("captured row count != number of arms");
                    for (int k = 0; k < K; ++k) {
                        const sample_rec rec = { (int32_t) n_prefix, k, s, tok[k], (int32_t) r, a + s, srec[k].p, srec[k].entropy };
                        fwrite(&rec, sizeof(rec), 1, f_samples);
                        for (size_t sl = 0; sl < layers.size(); ++sl) {
                            ggml_fp32_to_fp16_row(cap.buf[sl].data() + (size_t) k * n_embd, row16.data(), n_embd);
                            fwrite(row16.data(), sizeof(ggml_fp16_t), n_embd, f_acts[sl]);
                        }
                        n_samples++;
                    }
                }
            }
            cap.enabled = false;
            for (int k = 0; k < K; ++k) llama_memory_seq_rm(mem, k + 1, -1, -1);
            n_prefix++;
        }
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        fprintf(stderr, "rollout %zu/%zu: %lld prefixes, %lld samples, %.1f s/prefix\n",
                r + 1, rollouts.size(), (long long) n_prefix, (long long) n_samples, el / std::max<int64_t>(1, n_prefix));
    }

    fclose(f_steps);
    fclose(f_samples);
    for (FILE * f : f_acts) fclose(f);

    json meta;
    meta["model"] = model_path; meta["rollouts"] = rollouts_path;
    meta["levels"] = levels; meta["width"] = width; meta["prefix_lens"] = prefix_lens;
    meta["n_steps"] = n_steps; meta["n_capture"] = n_capture; meta["layers"] = layers;
    meta["n_arms"] = K; meta["arms"] = temp_arms ? "0 normal, 1..L band:level, L+1..2L temp:level" : "0 normal, 1..L band:level";
    meta["fallback"] = fallback_nearest ? "nearest" : "normal";
    meta["n_prefix"] = n_prefix; meta["n_samples"] = n_samples; meta["n_embd"] = n_embd; meta["seed"] = seed;
    meta["step_rec"] = "i32 prefix, i32 arm, i32 step, i32 token, f32 p, f32 entropy, f32 T, i32 band_hit, f32 band_mass[8]";
    meta["sample_rec"] = "i32 prefix, i32 arm, i32 step, i32 token, i32 seq, i32 pos, f32 p, f32 entropy";
    std::ofstream(out_dir + "/meta.json") << meta.dump(2) << "\n";
    fprintf(stderr, "homeo-band: %lld prefixes, %lld samples -> %s\n", (long long) n_prefix, (long long) n_samples, out_dir.c_str());

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
