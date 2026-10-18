// homeo-lean: data for the lean test (H3 of the Value Lens proposal, NOTES-homeostasis.md).
//
// Does the model give more probability to next tokens that lead to easier futures than real text
// supports? At positions t of held-out real text (fineweb ids), take the top --n-cand candidates for
// the next token under the model, branch the KV cache on each (seq_cp on a unified cache), and continue
// every branch --n-roll times for --horizon steps at temperature 1 with no truncation. Rollout r uses
// the same uniforms for every candidate at a position (common random numbers, inverse-CDF sampling).
// Recorded per candidate x: log p(x), the entropy H_k of the model's next-token distribution k steps
// after x (k = 1..horizon, averaged over rollouts), and per rollout the mean entropy and the mean
// surprisal of its sampled tokens. V(x) = -mean_k H_k is the future certainty after x.
// The fit (homeo/lean_fit.py) is q(x) ~ p(x)^a exp(b V(x)) over the candidates, against the actual next
// token: b < 0 means the model over-weights easy-future tokens (a lean toward predictability).
//
// No steering: the candidates are the model's own top predictions and the rollouts are plain samples.
//
// usage: homeo-lean -m model.gguf --ids fineweb_ids.bin --out lean.jsonl
//          [--n-windows 3000] [--n-tokens 170] [--pos-start 48] [--pos-step 24] [--n-cand 8] [--n-roll 4]
//          [--horizon 16] [--seed 7] [--pause-ms 0] [--threads N]
// Windows start at random offsets (seed). One JSON line per window; rerunning resumes after the last
// complete line. --pause-ms N sleeps N ms after each decode (quieter GPU); it is re-read from
// OUT.pause_ms before every window if that file exists.

#include "llama.h"
#include "ggml.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

static void die(const char * msg) {
    fprintf(stderr, "homeo-lean: %s\n", msg);
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

// log-softmax stats of one logit row: entropy, and the log-sum-exp
struct row_stats { double lse, entropy; };

static row_stats stats_of(const float * z, int n) {
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) m = std::max(m, z[i]);
    double s = 0.0, sz = 0.0;
    for (int i = 0; i < n; ++i) {
        const double e = std::exp((double) z[i] - m);
        s  += e;
        sz += e * z[i];
    }
    const double lse = m + std::log(s);
    return { lse, lse - sz / s };
}

// inverse-CDF sample at uniform u (vocab order); returns the token
static int sample_at(const float * z, int n, double lse, double u) {
    double c = 0.0;
    for (int i = 0; i < n; ++i) {
        c += std::exp((double) z[i] - lse);
        if (c >= u) return i;
    }
    return n - 1;
}

int main(int argc, char ** argv) {
    std::string model_path, ids_path, out_path;
    int n_windows = 3000, T = 170, pos_start = 48, pos_step = 24, N = 8, R = 4, K = 16, seed = 7, pause_ms = 0;
    int n_threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) die(("missing value for " + a).c_str()); return argv[++i]; };
        if      (a == "-m")            model_path = next();
        else if (a == "--ids")         ids_path   = next();
        else if (a == "--out")         out_path   = next();
        else if (a == "--n-windows")   n_windows  = atoi(next());
        else if (a == "--n-tokens")    T          = atoi(next());
        else if (a == "--pos-start")   pos_start  = atoi(next());
        else if (a == "--pos-step")    pos_step   = atoi(next());
        else if (a == "--n-cand")      N          = atoi(next());
        else if (a == "--n-roll")      R          = atoi(next());
        else if (a == "--horizon")     K          = atoi(next());
        else if (a == "--seed")        seed       = atoi(next());
        else if (a == "--pause-ms")    pause_ms   = atoi(next());
        else if (a == "--threads")     n_threads  = atoi(next());
        else die(("unknown argument " + a).c_str());
    }
    if (model_path.empty() || ids_path.empty() || out_path.empty()) {
        die("usage: homeo-lean -m model.gguf --ids ids.bin --out lean.jsonl [options]");
    }
    if (pos_start < 1 || pos_step < 1 || pos_start >= T - 1 || N < 2 || R < 1 || K < 1) die("bad sizes");
    std::vector<int> P;
    for (int t = pos_start; t <= T - 2; t += pos_step) P.push_back(t);
    const int NB = N * R; // branch sequences 1..NB

    // ---- token ids and window offsets ----
    std::vector<int32_t> ids;
    {
        std::ifstream f(ids_path, std::ios::binary | std::ios::ate);
        if (!f) die("cannot open --ids");
        const size_t n = (size_t) f.tellg() / sizeof(int32_t);
        ids.resize(n);
        f.seekg(0);
        f.read((char *) ids.data(), n * sizeof(int32_t));
    }
    if (ids.size() < (size_t) T + 1) die("--ids too short");
    std::vector<int64_t> offs(n_windows);
    {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int64_t> U(0, (int64_t) ids.size() - T - 1);
        for (auto & o : offs) o = U(rng);
    }

    // resume: count complete lines and cut a partial last line (crash mid-write)
    int done = 0;
    {
        std::ifstream f(out_path, std::ios::binary);
        std::string line;
        size_t good = 0, pos = 0;
        while (std::getline(f, line)) {
            pos += line.size() + 1;
            if (!line.empty() && line.back() == '}' && !f.eof()) { done++; good = pos; }
        }
        f.close();
        if (std::filesystem::exists(out_path) && std::filesystem::file_size(out_path) != good) {
            fprintf(stderr, "homeo-lean: cutting a partial last line\n");
            std::filesystem::resize_file(out_path, good);
        }
    }
    if (done >= n_windows) {
        fprintf(stderr, "homeo-lean: finished %d windows (nothing to do)\n", done);
        return 0;
    }

    // ---- model + context ----
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) die("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = T + NB * K + 64;
    cp.n_batch         = std::max(T, NB);
    cp.n_ubatch        = std::max(T, NB);
    cp.n_seq_max       = NB + 1;
    cp.kv_unified      = true; // shared prefix cells for the branches
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.no_perf         = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context init failed");
    llama_memory_t mem = llama_get_memory(ctx);
    llama_batch batch = llama_batch_init(std::max(T, NB), 0, 1);

    auto decode = [&](int n) {
        batch.n_tokens = n;
        if (llama_decode(ctx, batch) != 0) die("decode failed");
        if (pause_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms));
    };

    FILE * fo = fopen(out_path.c_str(), "a");
    if (!fo) die("cannot open --out");
    {
        json meta = { {"model", model_path}, {"ids", ids_path}, {"n_windows", n_windows}, {"n_tokens", T},
                      {"positions", P}, {"n_cand", N}, {"n_roll", R}, {"horizon", K}, {"seed", seed} };
        std::ofstream(out_path + ".meta.json") << meta.dump(1) << "\n";
    }
    fprintf(stderr, "homeo-lean: %d windows (%d done), %zu positions each, %d candidates x %d rollouts x %d steps\n",
            n_windows, done, P.size(), N, R, K);

    std::vector<float> zt(n_vocab);
    const auto t_start = std::chrono::steady_clock::now();
    for (int w = done; w < n_windows; ++w) {
        {
            std::ifstream pf(out_path + ".pause_ms");
            int v;
            if (pf >> v && v >= 0 && v != pause_ms) {
                pause_ms = v;
                fprintf(stderr, "homeo-lean: pause %d ms per decode\n", pause_ms);
            }
        }
        const int32_t * tok = ids.data() + offs[w];
        llama_memory_clear(mem, true);

        // prefill the window in seq 0 with logits everywhere (for the past-surprisal covariate)
        for (int i = 0; i < T; ++i) {
            batch.token[i] = tok[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = 1;
        }
        decode(T);
        std::vector<float> surp(T, 0.0f), ent(T, 0.0f); // surp[i] = -log p(tok[i] | tok[<i]), ent[i] = H(p_i)
        std::vector<std::vector<float>> zrows_pos(P.size());
        parallel_for(T, n_threads, [&](int i) {
            const float * z = llama_get_logits_ith(ctx, i);
            const row_stats st = stats_of(z, n_vocab);
            ent[i] = (float) st.entropy;
            if (i + 1 < T) surp[i + 1] = (float) (st.lse - z[tok[i + 1]]);
        });
        for (size_t pi = 0; pi < P.size(); ++pi) {
            const float * z = llama_get_logits_ith(ctx, P[pi]);
            zrows_pos[pi].assign(z, z + n_vocab);
        }

        json wj = { {"w", w}, {"offset", offs[w]}, {"positions", json::array()} };
        for (size_t pi = 0; pi < P.size(); ++pi) {
            const int t = P[pi];
            std::memcpy(zt.data(), zrows_pos[pi].data(), sizeof(float) * n_vocab);
            const row_stats st = stats_of(zt.data(), n_vocab);
            std::vector<int> idx(n_vocab);
            for (int v = 0; v < n_vocab; ++v) idx[v] = v;
            std::partial_sort(idx.begin(), idx.begin() + N, idx.end(), [&](int a, int b) { return zt[a] > zt[b]; });

            // uniforms shared by all candidates: u[r][k]
            std::mt19937_64 rng((uint64_t) seed * 1000003ull + (uint64_t) w * 1009ull + (uint64_t) t);
            std::uniform_real_distribution<double> U01(0.0, 1.0);
            std::vector<double> u((size_t) R * K);
            for (auto & x : u) x = U01(rng);

            for (int b = 0; b < NB; ++b) llama_memory_seq_cp(mem, 0, b + 1, 0, t + 1);
            std::vector<int>    cur(NB);
            std::vector<double> Hsum((size_t) NB * K, 0.0), Ssum(NB, 0.0);
            for (int b = 0; b < NB; ++b) cur[b] = idx[b / R];
            for (int k = 0; k < K; ++k) {
                for (int b = 0; b < NB; ++b) {
                    batch.token[b] = cur[b]; batch.pos[b] = t + 1 + k; batch.n_seq_id[b] = 1;
                    batch.seq_id[b][0] = b + 1; batch.logits[b] = 1;
                }
                decode(NB);
                parallel_for(NB, n_threads, [&](int b) {
                    const float * z = llama_get_logits_ith(ctx, b);
                    const row_stats s2 = stats_of(z, n_vocab);
                    const int y = sample_at(z, n_vocab, s2.lse, u[(size_t) (b % R) * K + k]);
                    Hsum[(size_t) b * K + k] = s2.entropy;
                    Ssum[b] += s2.lse - z[y];
                    cur[b] = y;
                });
            }
            for (int b = 0; b < NB; ++b) llama_memory_seq_rm(mem, b + 1, -1, -1);

            double past = 0.0;
            for (int i = std::max(1, t - 31); i <= t; ++i) past += surp[i];
            past /= (t - std::max(1, t - 31) + 1);
            // H_next: entropy after the actual token from the prefill; the actual token's branch must match its
            // first-step entropy up to batch-shape numerics (consistency check)
            json pj = { {"t", t}, {"actual", tok[t + 1]}, {"H", st.entropy}, {"H_next", ent[t + 1]}, {"surp_t", surp[t]},
                        {"past32", past}, {"cands", json::array()} };
            for (int c = 0; c < N; ++c) {
                std::vector<double> Hk(K, 0.0), Hr(R, 0.0), Sr(R, 0.0);
                for (int r = 0; r < R; ++r) {
                    const int b = c * R + r;
                    for (int k = 0; k < K; ++k) {
                        Hk[k] += Hsum[(size_t) b * K + k] / R;
                        Hr[r] += Hsum[(size_t) b * K + k] / K;
                    }
                    Sr[r] = Ssum[b] / K;
                }
                auto r4 = [](double x) { return std::round(x * 1e4) / 1e4; };
                json cj = { {"tok", idx[c]}, {"lp", r4(zt[idx[c]] - st.lse)}, {"Hk", json::array()}, {"Hr", json::array()},
                            {"Sr", json::array()} };
                for (double x : Hk) cj["Hk"].push_back(r4(x));
                for (double x : Hr) cj["Hr"].push_back(r4(x));
                for (double x : Sr) cj["Sr"].push_back(r4(x));
                pj["cands"].push_back(cj);
            }
            wj["positions"].push_back(pj);
        }
        const std::string line = wj.dump() + "\n";
        if (fwrite(line.data(), 1, line.size(), fo) != line.size() || fflush(fo) != 0) die("write failed");
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        fprintf(stderr, "homeo-lean: window %d done (%.1f s per window this run)\n", w, el / (w - done + 1));
    }
    fclose(fo);
    fprintf(stderr, "homeo-lean: finished %d windows\n", n_windows);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
