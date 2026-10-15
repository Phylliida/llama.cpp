// homeo-capture: residual capture for the surprise-probe experiment (S1 in
// NOTES-homeostasis.md): does the base model represent "the previous token
// was unlikely"?
//
// Teacher-forces each rollout (gen_rollouts.py JSONL: prompt_ids + text) with
// logits at every position and records, for every position t:
//   surprisal     -ln p_{t-1}(x_t) under the model's own previous-position distribution
//   entropy_prev  H(p_{t-1})
//   rank_prev     rank of x_t under p_{t-1} (0 = argmax)
//   entropy       H(p_t), the next-token entropy after seeing x_t
// Block outputs l_out-N are captured at
//   natural positions: every --stride-th position from --skip on
//   forced positions:  --n-forced contexts per rollout. The prefix is branched
//     (seq_cp on a unified KV cache) and one random token from each log10-p
//     bin is decoded at the same position in its own sequence, plus the
//     natural token (kind 2) as a consistency check against the natural pass.
//
// usage: homeo-capture -m model.gguf --rollouts rollouts.jsonl --out DIR
//          [--layers 0,4,...] [--stride 4] [--skip 32] [--n-forced 16]
//          [--max-len 2100] [--n-rollouts N] [--seed 1] [--ubatch 512]
//
// outputs in DIR (little endian):
//   labels.bin    label_rec per (rollout, position)
//   samples.bin   sample_rec per captured row
//   acts_LNN.f16  [n_samples, n_embd] f16, row i <-> samples.bin record i
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

struct label_rec {
    int32_t seq;
    int32_t pos;
    int32_t token;
    int32_t rank_prev;
    float   surprisal;
    float   entropy_prev;
    float   entropy;
    float   pad;
};
static_assert(sizeof(label_rec) == 32, "label_rec layout");

struct sample_rec {
    int32_t seq;
    int32_t pos;
    int32_t token;
    int32_t kind; // 0 natural, 1 forced, 2 forced natural token (consistency check)
    int32_t bin;  // log10-p bin of the token under p_{pos-1}
    int32_t rank_prev;
    float   surprisal;
    float   entropy_prev;
    float   entropy;
    float   pad[3];
};
static_assert(sizeof(sample_rec) == 48, "sample_rec layout");

// log10 p bin edges, descending: bin 0 is p >= 10^-0.5, bin 11 is p < 1e-10
static const float BIN_EDGES[] = { -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -10.0f };
static const int   N_BINS      = 12;

static int lp10_bin(float lp10) {
    int b = 0;
    while (b < N_BINS - 1 && lp10 < BIN_EDGES[b]) {
        b++;
    }
    return b;
}

static void die(const char * msg) {
    fprintf(stderr, "homeo-capture: %s\n", msg);
    exit(1);
}

struct row_stats {
    float lse;
    float entropy;
};

static row_stats stats_of(const float * z, int n) {
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) {
        m = std::max(m, z[i]);
    }
    double s = 0.0, sz = 0.0;
    for (int i = 0; i < n; ++i) {
        const double e = std::exp(z[i] - m);
        s  += e;
        sz += e * z[i];
    }
    const double lse = m + std::log(s);
    return { (float) lse, (float) (lse - sz / s) };
}

static int32_t rank_of(const float * z, int n, int tok) {
    const float zt = z[tok];
    int32_t r = 0;
    for (int i = 0; i < n; ++i) {
        r += z[i] > zt;
    }
    return r;
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

// ---- l_out capture via the scheduler eval callback ----

struct capture {
    std::unordered_map<std::string, int> slot_of; // "l_out-N" -> slot
    std::vector<std::vector<float>>      buf;     // per slot: [n_embd, rows] of the last decode
    std::vector<int64_t>                 rows;
    int64_t                              n_embd = 0;
};

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    auto * c  = (capture *) ud;
    auto   it = c->slot_of.find(t->name);
    if (it == c->slot_of.end()) {
        return !ask; // not ours: don't observe (ask), keep computing (observe)
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

static void decode_checked(llama_context * ctx, capture & cap, const llama_batch & batch) {
    std::fill(cap.rows.begin(), cap.rows.end(), 0);
    if (llama_decode(ctx, batch) != 0) {
        die("llama_decode failed");
    }
    for (int64_t r : cap.rows) {
        if (r != batch.n_tokens) {
            die("captured row count != batch size (ubatch split or output pruning?)");
        }
    }
}

int main(int argc, char ** argv) {
    std::string model_path, rollouts_path, out_dir;
    std::string layers_str;
    int  stride     = 4;
    int  skip       = 32;
    int  n_forced   = 16;
    int  max_len    = 2100;
    int  n_rollouts = -1;
    int  seed       = 1;
    int  ubatch     = 512;
    int  n_threads  = (int) std::max(1u, std::thread::hardware_concurrency() / 2);

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) die(("missing value for " + a).c_str()); return argv[++i]; };
        if      (a == "-m")           model_path    = next();
        else if (a == "--rollouts")   rollouts_path = next();
        else if (a == "--out")        out_dir       = next();
        else if (a == "--layers")     layers_str    = next();
        else if (a == "--stride")     stride        = atoi(next());
        else if (a == "--skip")       skip          = atoi(next());
        else if (a == "--n-forced")   n_forced      = atoi(next());
        else if (a == "--max-len")    max_len       = atoi(next());
        else if (a == "--n-rollouts") n_rollouts    = atoi(next());
        else if (a == "--seed")       seed          = atoi(next());
        else if (a == "--ubatch")     ubatch        = atoi(next());
        else if (a == "--threads")    n_threads     = atoi(next());
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || rollouts_path.empty() || out_dir.empty()) {
        fprintf(stderr, "usage: homeo-capture -m model.gguf --rollouts rollouts.jsonl --out DIR [options]\n");
        return 1;
    }
    if (skip < 1 || stride < 1) die("--skip and --stride must be >= 1");

    // ---- rollouts ----
    struct rollout { std::vector<int32_t> prompt; std::string text; int doc_idx, rollout_idx; };
    std::vector<rollout> rollouts;
    {
        std::ifstream f(rollouts_path);
        if (!f) die("cannot open rollouts file");
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const json j = json::parse(line);
            rollouts.push_back({ j.at("prompt_ids").get<std::vector<int32_t>>(), j.at("text").get<std::string>(),
                                 j.value("doc_idx", -1), j.value("rollout_idx", -1) });
        }
    }
    if (n_rollouts >= 0 && (size_t) n_rollouts < rollouts.size()) rollouts.resize(n_rollouts);
    fprintf(stderr, "homeo-capture: %zu rollouts\n", rollouts.size());

    // ---- model + context ----
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) die("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);

    std::vector<int> layers;
    if (layers_str.empty()) {
        for (int il = 0; il < n_layer; il += 4) layers.push_back(il);
        if (layers.back() != n_layer - 1) layers.push_back(n_layer - 1);
    } else {
        char * s = strdup(layers_str.c_str());
        for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) layers.push_back(atoi(t));
        free(s);
    }
    for (int il : layers) if (il < 0 || il >= n_layer) die("--layers out of range");

    capture cap;
    cap.n_embd = n_embd;
    cap.buf.resize(layers.size());
    cap.rows.resize(layers.size());
    for (size_t s = 0; s < layers.size(); ++s) {
        cap.slot_of["l_out-" + std::to_string(layers[s])] = (int) s;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = max_len + 256;
    cp.n_batch           = ubatch;
    cp.n_ubatch          = ubatch;
    cp.n_seq_max         = N_BINS + 2; // natural seq + one branch per bin + natural-token branch
    cp.kv_unified        = true;       // shared prefix cells; split_simple keeps batch order
    cp.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval           = cb_eval;
    cp.cb_eval_user_data = &cap;
    cp.n_threads         = n_threads;
    cp.n_threads_batch   = n_threads;
    cp.no_perf           = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context init failed");
    llama_memory_t mem = llama_get_memory(ctx);

    // ---- outputs ----
    if (system(("mkdir -p '" + out_dir + "'").c_str()) != 0) die("mkdir failed");
    FILE * f_labels  = fopen((out_dir + "/labels.bin").c_str(), "wb");
    FILE * f_samples = fopen((out_dir + "/samples.bin").c_str(), "wb");
    if (!f_labels || !f_samples) die("cannot open output files");
    std::vector<FILE *> f_acts;
    for (int il : layers) {
        char name[64];
        snprintf(name, sizeof(name), "/acts_L%02d.f16", il);
        FILE * f = fopen((out_dir + name).c_str(), "wb");
        if (!f) die("cannot open acts file");
        f_acts.push_back(f);
    }

    std::vector<ggml_fp16_t> row16(n_embd);
    int64_t n_samples = 0, n_labels = 0;
    int64_t n_kind[3] = { 0, 0, 0 };
    auto write_sample = [&](const sample_rec & rec, int row) {
        fwrite(&rec, sizeof(rec), 1, f_samples);
        for (size_t s = 0; s < layers.size(); ++s) {
            ggml_fp32_to_fp16_row(cap.buf[s].data() + (size_t) row * n_embd, row16.data(), n_embd);
            fwrite(row16.data(), sizeof(ggml_fp16_t), n_embd, f_acts[s]);
        }
        n_samples++;
        n_kind[rec.kind]++;
    };

    // consistency: natural-token branch vs natural pass at the same position
    std::vector<double> chk_sum(layers.size(), 0.0), chk_max(layers.size(), 0.0);
    double chk_dH_max = 0.0;
    int64_t chk_n = 0;

    std::mt19937_64 rng(seed);
    llama_batch batch = llama_batch_init(ubatch, 0, 1);
    std::vector<const float *> zrows;
    std::vector<float> zprev(n_vocab);
    const float LN10 = std::log(10.0f);
    const auto t_start = std::chrono::steady_clock::now();
    int64_t n_tok_total = 0;

    for (size_t r = 0; r < rollouts.size(); ++r) {
        // tokens: prompt ids + generated text (parse_special: rollouts may contain doc-restart specials)
        std::vector<int32_t> toks = rollouts[r].prompt;
        {
            const std::string & text = rollouts[r].text;
            std::vector<llama_token> tt(text.size() + 16);
            int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tt.data(), (int32_t) tt.size(),
                                       /*add_special=*/false, /*parse_special=*/true);
            if (n < 0) die("tokenize failed");
            toks.insert(toks.end(), tt.begin(), tt.begin() + n);
        }
        if ((int) toks.size() > max_len) toks.resize(max_len);
        const int len = (int) toks.size();
        if (len <= skip + 1) continue;

        std::vector<label_rec> lab(len);
        for (int i = 0; i < len; ++i) {
            lab[i] = { (int32_t) r, i, toks[i], -1, NAN, NAN, NAN, 0.0f };
        }

        // forced positions: distinct, in [skip, len)
        std::vector<int> fpos;
        {
            std::vector<int> cand;
            for (int i = skip; i < len; ++i) cand.push_back(i);
            std::shuffle(cand.begin(), cand.end(), rng);
            cand.resize(std::min<size_t>(cand.size(), (size_t) n_forced));
            std::sort(cand.begin(), cand.end());
            fpos = cand;
        }

        llama_memory_clear(mem, true);
        std::vector<std::vector<float>> chk_rows;
        float chk_H   = NAN;
        int   chk_pos = -1;
        size_t fi = 0;
        int cur = 0;

        while (cur < len) {
            const int stop = fi < fpos.size() ? fpos[fi] : len;
            const int end  = std::min(cur + ubatch, stop);
            const int n    = end - cur;

            batch.n_tokens = n;
            for (int j = 0; j < n; ++j) {
                batch.token[j]     = toks[cur + j];
                batch.pos[j]       = cur + j;
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = 1;
            }
            decode_checked(ctx, cap, batch);

            zrows.resize(n);
            for (int j = 0; j < n; ++j) zrows[j] = llama_get_logits_ith(ctx, j);
            parallel_for(n, n_threads, [&](int j) {
                const int   i  = cur + j;
                const float * z = zrows[j];
                const row_stats st = stats_of(z, n_vocab);
                lab[i].entropy = st.entropy;
                if (i + 1 < len) {
                    lab[i + 1].entropy_prev = st.entropy;
                    lab[i + 1].surprisal    = st.lse - z[toks[i + 1]];
                    lab[i + 1].rank_prev    = rank_of(z, n_vocab, toks[i + 1]);
                }
            });

            if (chk_pos == cur) {
                for (size_t s = 0; s < layers.size(); ++s) {
                    const float * a = chk_rows[s].data();
                    const float * b = cap.buf[s].data(); // row 0 = position cur
                    double d2 = 0.0, b2 = 0.0;
                    for (int e = 0; e < n_embd; ++e) { const double d = a[e] - b[e]; d2 += d * d; b2 += (double) b[e] * b[e]; }
                    const double rel = std::sqrt(d2 / std::max(b2, 1e-30));
                    chk_sum[s] += rel;
                    chk_max[s]  = std::max(chk_max[s], rel);
                }
                chk_dH_max = std::max(chk_dH_max, (double) std::fabs(chk_H - lab[cur].entropy));
                chk_n++;
                chk_pos = -1;
            }

            for (int j = 0; j < n; ++j) {
                const int i = cur + j;
                if (i < skip || (i - skip) % stride != 0) continue;
                const label_rec & L = lab[i];
                sample_rec rec = {};
                rec.seq = (int32_t) r; rec.pos = i; rec.token = L.token; rec.kind = 0;
                rec.bin = lp10_bin(-L.surprisal / LN10);
                rec.rank_prev = L.rank_prev; rec.surprisal = L.surprisal;
                rec.entropy_prev = L.entropy_prev; rec.entropy = L.entropy;
                write_sample(rec, j);
            }

            const float * zlast = zrows[n - 1]; // p_{end-1}: predicts position end
            cur = end;

            if (cur == stop && stop < len) {
                // ---- branch at t: one random token per log10-p bin + the natural token ----
                const int t = stop;
                std::memcpy(zprev.data(), zlast, sizeof(float) * n_vocab);
                const row_stats st = stats_of(zprev.data(), n_vocab);

                std::vector<int>     pick(N_BINS, -1);
                std::vector<int64_t> cnt(N_BINS, 0);
                for (int v = 0; v < n_vocab; ++v) {
                    if (v == toks[t] || llama_vocab_is_control(vocab, v) || llama_vocab_is_eog(vocab, v)) continue;
                    const int b = lp10_bin((zprev[v] - st.lse) / LN10);
                    if (rng() % (uint64_t) (++cnt[b]) == 0) pick[b] = v; // reservoir sample
                }
                struct branch { int tok, kind; };
                std::vector<branch> br;
                for (int b = 0; b < N_BINS; ++b) if (pick[b] >= 0) br.push_back({ pick[b], 1 });
                br.push_back({ toks[t], 2 });

                const int K = (int) br.size();
                batch.n_tokens = K;
                for (int k = 0; k < K; ++k) {
                    llama_memory_seq_cp(mem, 0, k + 1, 0, t);
                    batch.token[k]     = br[k].tok;
                    batch.pos[k]       = t;
                    batch.n_seq_id[k]  = 1;
                    batch.seq_id[k][0] = k + 1;
                    batch.logits[k]    = 1;
                }
                decode_checked(ctx, cap, batch);

                for (int k = 0; k < K; ++k) {
                    const float * z = llama_get_logits_ith(ctx, k);
                    sample_rec rec = {};
                    rec.seq = (int32_t) r; rec.pos = t; rec.token = br[k].tok; rec.kind = br[k].kind;
                    rec.surprisal    = st.lse - zprev[br[k].tok];
                    rec.bin          = lp10_bin(-rec.surprisal / LN10);
                    rec.rank_prev    = rank_of(zprev.data(), n_vocab, br[k].tok);
                    rec.entropy_prev = st.entropy;
                    rec.entropy      = stats_of(z, n_vocab).entropy;
                    write_sample(rec, k);
                    if (br[k].kind == 2) {
                        chk_rows.assign(layers.size(), std::vector<float>(n_embd));
                        for (size_t s = 0; s < layers.size(); ++s) {
                            std::memcpy(chk_rows[s].data(), cap.buf[s].data() + (size_t) k * n_embd, sizeof(float) * n_embd);
                        }
                        chk_H   = rec.entropy;
                        chk_pos = t;
                    }
                }
                for (int k = 0; k < K; ++k) {
                    llama_memory_seq_rm(mem, k + 1, -1, -1);
                }
                fi++;
            }
        }

        fwrite(lab.data(), sizeof(label_rec), lab.size(), f_labels);
        n_labels    += len;
        n_tok_total += len;
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        fprintf(stderr, "rollout %zu/%zu len=%d samples=%lld (nat %lld forced %lld) %.0f tok/s\n",
                r + 1, rollouts.size(), len, (long long) n_samples, (long long) n_kind[0],
                (long long) (n_kind[1] + n_kind[2]), n_tok_total / el);
    }

    fclose(f_labels);
    fclose(f_samples);
    for (FILE * f : f_acts) fclose(f);

    json meta;
    meta["model"]       = model_path;
    meta["rollouts"]    = rollouts_path;
    meta["n_rollouts"]  = rollouts.size();
    meta["n_vocab"]     = n_vocab;
    meta["n_embd"]      = n_embd;
    meta["n_layer"]     = n_layer;
    meta["layers"]      = layers;
    meta["n_samples"]   = n_samples;
    meta["n_labels"]    = n_labels;
    meta["n_kind"]      = { n_kind[0], n_kind[1], n_kind[2] };
    meta["stride"]      = stride;
    meta["skip"]        = skip;
    meta["n_forced"]    = n_forced;
    meta["max_len"]     = max_len;
    meta["seed"]        = seed;
    meta["bin_edges_log10p"] = std::vector<float>(std::begin(BIN_EDGES), std::end(BIN_EDGES));
    meta["label_rec"]   = "i32 seq, i32 pos, i32 token, i32 rank_prev, f32 surprisal, f32 entropy_prev, f32 entropy, f32 pad";
    meta["sample_rec"]  = "i32 seq, i32 pos, i32 token, i32 kind, i32 bin, i32 rank_prev, f32 surprisal, f32 entropy_prev, f32 entropy, f32 pad[3]";
    json chk;
    chk["n"] = chk_n;
    chk["entropy_abs_diff_max"] = chk_dH_max;
    for (size_t s = 0; s < layers.size(); ++s) {
        chk["rel_err_mean"].push_back(chk_n ? chk_sum[s] / chk_n : 0.0);
        chk["rel_err_max"].push_back(chk_max[s]);
    }
    meta["consistency"] = chk;
    std::ofstream(out_dir + "/meta.json") << meta.dump(2) << "\n";

    fprintf(stderr, "homeo-capture: %lld samples, %lld labels -> %s\n", (long long) n_samples, (long long) n_labels, out_dir.c_str());
    fprintf(stderr, "consistency (natural-token branch vs natural pass, n=%lld): max |dH| %.2e\n", (long long) chk_n, chk_dH_max);
    for (size_t s = 0; s < layers.size(); ++s) {
        fprintf(stderr, "  L%02d rel err mean %.2e max %.2e\n", layers[s], chk_n ? chk_sum[s] / chk_n : 0.0, chk_max[s]);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
