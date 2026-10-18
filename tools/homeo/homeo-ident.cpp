// homeo-ident: closed-loop system identification of the layer-L state (plan B in NOTES-homeostasis.md).
//
// At a fixed layer, the state of a later position depends on earlier positions only through the tokens: a push
// at the output of block L reaches later layer-L states only by changing what the model writes. So for random
// probe directions u in a K-dim subspace (rows of --basis), steer with +a*u and -a*u after block L from the last
// prefix token on (a step input: the prefix itself is decoded once, unsteered, and shared by every arm through
// the KV cache; a = dose * mean |h_L|), sample --n-gen tokens per prefix at T=1 with common random numbers (one
// uniform per prefix and step, shared by all arms), and record the PRE-steer layer-L state of every generated
// position (l_out - a*u). The difference to the unsteered arm, projected on the basis, is
// the token-mediated response to the push. Fit offline (homeo/ident_fit.py): response ~ M(lag) z, z the probe
// coordinates. Directions where M pushes back (negative eigenvalues of its symmetric part) are candidate
// regulated variables; positive ones are amplified.
//
// Lag k = 0 is the last prefix token (no generated token yet), so its response must be exactly 0.
// Humane endpoints (Three Rs protocol) as in homeo-loop: a sequence stops on a loop or incoherence; its later
// lags are NaN. Every --log-every arms, at the end, and on SIGINT/SIGTERM (which stop after the current arm) a
// row goes to --log (welfare_log.csv).
// Prefixes: held-out rollouts (--rollouts, doc_idx % 4 == 0, as homeo-loop), or windows of a token-id stream
// (--ids, int32; window i starts at --ids-offset + i * --ids-stride).
//
// Output (resumable): OUT.bin, fixed-size records in this order: the unsteered arm, then for each dose, each
// probe, sign + then -. Record: f16 dproj [NP][G][K] (basis projection of h_pre - h_pre_unsteered),
// f32 dnorm [NP][G] (|h_pre - h_pre_unsteered|), f32 ent [NP][G] (entropy of the next-token distribution),
// i32 tok [NP][G], i32 first_div [NP] (first step whose token differs from the unsteered arm, -1 = none),
// i32 endpoint [NP] (0 none, 1 loop, 2 incoherence). OUT.meta.json describes the run, OUT.z.f32 holds the
// probe coordinates [P][K]. A rerun with the same arguments skips the records already in OUT.bin.
//
// usage: homeo-ident -m model.gguf (--rollouts rollouts.jsonl | --ids ids.bin) --basis basis.f32 --k K --layer L
//          --out PREFIX [--n-probe 64] [--doses 0.25] [--n-prefix 16] [--prefix-len 256] [--n-gen 48] [--seed 1]
//          [--probe-seed 7] [--ids-offset 0] [--ids-stride 100003] [--pause-ms 0] [--log welfare_log.csv]
//          [--log-every 16] [--note text]
// --pause-ms is re-read from PREFIX.pause_ms (if it exists) before every arm, to change the duty cycle live.

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
#include <ctime>
#include <fstream>
#include <csignal>
#include <random>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

static void die(const char * msg) {
    fprintf(stderr, "homeo-ident: %s\n", msg);
    exit(1);
}

template <class F>
static void parallel_for(int n, int n_threads, const F & f) {
    const int chunk = (n + n_threads - 1) / n_threads;
    std::vector<std::thread> th;
    for (int t = 0; t < n_threads; ++t) {
        const int a = t * chunk, b = std::min(n, a + chunk);
        if (a >= b) break;
        th.emplace_back([a, b, &f] { for (int i = a; i < b; ++i) f(i); });
    }
    for (auto & x : th) x.join();
}

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= s.size()) {
        const size_t b = std::min(s.find(',', a), s.size());
        if (b > a) out.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return out;
}

static std::string base_name(const std::string & p) {
    std::string s = p.substr(p.find_last_of('/') + 1);
    const size_t dot = s.find_last_of('.');
    return dot == std::string::npos ? s : s.substr(0, dot);
}

static long long file_size(const std::string & p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 ? (long long) st.st_size : -1;
}

static void log_softmax(const float * z, float * out, int n) {
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) m = std::max(m, z[i]);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp(z[i] - m);
    const float lse = m + (float) std::log(s);
    for (int i = 0; i < n; ++i) out[i] = z[i] - lse;
}

// smallest period p whose repeat covers the last max(3p, min_span) tokens; 0 if none
static int find_loop(const std::vector<int32_t> & g, int max_p, int min_span) {
    const int n = (int) g.size();
    for (int p = 1; p <= max_p; ++p) {
        const int span = std::max(3 * p, min_span);
        if (n < span) break;
        bool ok = true;
        for (int i = n - span + p; i < n && ok; ++i) ok = g[i] == g[i - p];
        if (ok) return p;
    }
    return 0;
}

static std::string csv_q(const std::string & s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"') o += '"'; o += c; }
    return o + "\"";
}

// ---- l_out capture ----

struct capture {
    std::string        name;
    std::vector<float> buf;
    int64_t            rows    = 0;
    int64_t            n_embd  = 0;
    bool               enabled = false;
};

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    auto * c = (capture *) ud;
    if (!c->enabled || c->name != t->name) return !ask;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd || !ggml_is_contiguous(t)) die("unexpected l_out tensor layout");
    c->rows = ggml_nrows(t);
    c->buf.resize(c->rows * c->n_embd);
    ggml_backend_tensor_get(t, c->buf.data(), 0, ggml_nbytes(t));
    return true;
}

// one arm's result
struct arm_out {
    std::vector<float>   h;        // [NP][G][n_embd] pre-steer states (NaN after an endpoint)
    std::vector<float>   ent;      // [NP][G]
    std::vector<int32_t> tok;      // [NP][G] (-1 after an endpoint)
    std::vector<int32_t> endpoint; // [NP]
    int64_t n_tok = 0;
};

int main(int argc, char ** argv) {
    std::string model_path, rollouts_path, ids_path, out_prefix, basis_path, log_path, note;
    long long ids_offset = 0, ids_stride = 100003;
    std::string doses_str = "0.25";
    int layer = -1, K = 0, n_probe = 64, n_prefix = 16, plen = 256, n_gen = 48, seed = 1, probe_seed = 7, ngl = 999;
    int loop_max_p = 16, loop_min_span = 12, incoh_run = 16, ubatch = 512, pause_ms = 0, log_every = 16;
    float incoh_h = 6.0f;
    int n_threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) die(("missing value for " + a).c_str()); return argv[++i]; };
        if      (a == "-m")                model_path    = next();
        else if (a == "--rollouts")        rollouts_path = next();
        else if (a == "--ids")             ids_path      = next();
        else if (a == "--ids-offset")      ids_offset    = atoll(next());
        else if (a == "--ids-stride")      ids_stride    = atoll(next());
        else if (a == "--out")             out_prefix    = next();
        else if (a == "--basis")           basis_path    = next();
        else if (a == "--k")               K             = atoi(next());
        else if (a == "--layer")           layer         = atoi(next());
        else if (a == "--n-probe")         n_probe       = atoi(next());
        else if (a == "--doses")           doses_str     = next();
        else if (a == "--n-prefix")        n_prefix      = atoi(next());
        else if (a == "--prefix-len")      plen          = atoi(next());
        else if (a == "--n-gen")           n_gen         = atoi(next());
        else if (a == "--seed")            seed          = atoi(next());
        else if (a == "--probe-seed")      probe_seed    = atoi(next());
        else if (a == "--ngl")             ngl           = atoi(next());
        else if (a == "--threads")         n_threads     = atoi(next());
        else if (a == "--loop-max-period") loop_max_p    = atoi(next());
        else if (a == "--loop-min-span")   loop_min_span = atoi(next());
        else if (a == "--incoh-h")         incoh_h       = (float) atof(next());
        else if (a == "--incoh-run")       incoh_run     = atoi(next());
        else if (a == "--pause-ms")        pause_ms      = atoi(next());
        else if (a == "--log")             log_path      = next();
        else if (a == "--log-every")       log_every     = atoi(next());
        else if (a == "--note")            note          = next();
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || rollouts_path.empty() == ids_path.empty() || out_prefix.empty() || basis_path.empty() || layer < 1 || K < 1) {
        fprintf(stderr, "usage: homeo-ident -m model.gguf (--rollouts rollouts.jsonl | --ids ids.bin) --basis basis.f32 --k K --layer L --out PREFIX [options]\n");
        return 1;
    }
    std::vector<float> doses;
    for (auto & s : split(doses_str)) doses.push_back((float) atof(s.c_str()));
    const int G = n_gen;

    // ---- held-out prefixes (doc_idx % 4 == 0, as homeo-loop) ----
    struct prefix { std::vector<int32_t> toks; int doc; int rollout; };
    std::vector<prefix> pre;
    struct ro { std::vector<int32_t> prompt; std::string text; int doc; int rollout; };
    std::vector<ro> rolls;
    if (!rollouts_path.empty()) {
        std::ifstream f(rollouts_path);
        if (!f) die("cannot open rollouts file");
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const json j = json::parse(line);
            const int doc = j.value("doc_idx", 0);
            if (doc % 4 != 0) continue;
            rolls.push_back({ j.at("prompt_ids").get<std::vector<int32_t>>(), j.at("text").get<std::string>(), doc, j.value("rollout_idx", 0) });
        }
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) die("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    if (layer >= n_layer - 1) die("--layer must be < n_layer - 1");

    if (!ids_path.empty()) {
        std::ifstream f(ids_path, std::ios::binary | std::ios::ate);
        if (!f) die("cannot open --ids");
        std::vector<int32_t> ids((size_t) f.tellg() / sizeof(int32_t));
        f.seekg(0);
        f.read((char *) ids.data(), ids.size() * sizeof(int32_t));
        for (int i = 0; i < n_prefix; ++i) {
            const long long o = ids_offset + (long long) i * ids_stride;
            if (o + plen > (long long) ids.size()) die("--ids stream too short for --n-prefix windows");
            pre.push_back({ std::vector<int32_t>(ids.begin() + o, ids.begin() + o + plen), (int) (o / 1000), i });
        }
    }
    for (auto & r : rolls) {
        if ((int) pre.size() >= n_prefix) break;
        std::vector<int32_t> toks = r.prompt;
        std::vector<llama_token> tt(r.text.size() + 16);
        const int32_t n = llama_tokenize(vocab, r.text.c_str(), (int32_t) r.text.size(), tt.data(), (int32_t) tt.size(), false, true);
        if (n < 0) die("tokenize failed");
        toks.insert(toks.end(), tt.begin(), tt.begin() + n);
        if ((int) toks.size() < plen) continue;
        toks.resize(plen);
        pre.push_back({ toks, r.doc, r.rollout });
    }
    const int NP = (int) pre.size();
    if (NP < n_prefix) die("not enough held-out prefixes of --prefix-len tokens");

    // ---- basis (orthonormal rows) and probes ----
    std::vector<float> Q((size_t) K * n_embd);
    {
        FILE * f = fopen(basis_path.c_str(), "rb");
        if (!f || fread(Q.data(), sizeof(float), Q.size(), f) != Q.size()) die("cannot read --basis (f32 [>=K][n_embd])");
        fclose(f);
        double worst = 0.0;
        for (int a = 0; a < K; ++a)
            for (int b = a; b < K; ++b) {
                double d = 0.0;
                for (int e = 0; e < n_embd; ++e) d += (double) Q[(size_t) a * n_embd + e] * Q[(size_t) b * n_embd + e];
                worst = std::max(worst, std::fabs(d - (a == b ? 1.0 : 0.0)));
            }
        if (worst > 1e-3) die("basis rows are not orthonormal");
    }
    std::vector<float> Z((size_t) n_probe * K);
    std::vector<std::vector<float>> U(n_probe, std::vector<float>(n_embd, 0.0f));
    for (int p = 0; p < n_probe; ++p) {
        std::mt19937_64 rng((uint64_t) probe_seed * 1000003ull + (uint64_t) p);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        double s2 = 0.0;
        for (int k = 0; k < K; ++k) { Z[(size_t) p * K + k] = nd(rng); s2 += (double) Z[(size_t) p * K + k] * Z[(size_t) p * K + k]; }
        for (int k = 0; k < K; ++k) Z[(size_t) p * K + k] = (float) (Z[(size_t) p * K + k] / std::sqrt(s2));
        for (int k = 0; k < K; ++k)
            for (int e = 0; e < n_embd; ++e) U[p][e] += Z[(size_t) p * K + k] * Q[(size_t) k * n_embd + e];
    }

    // ---- context ----
    capture cap;
    cap.name   = "l_out-" + std::to_string(layer);
    cap.n_embd = n_embd;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = NP * (plen + G) + 512;
    cp.n_batch           = ubatch;
    cp.n_ubatch          = ubatch;
    cp.n_seq_max         = 2 * NP; // 0..NP-1 arms, NP..2NP-1 the shared unsteered prefixes
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
    llama_batch batch = llama_batch_init(std::max(ubatch, 2 * NP), 0, 1);

    // mean residual norm at the layer (unsteered prefixes, skipping the first 16 positions)
    if (llama_set_adapter_cvec(ctx, nullptr, 0, n_embd, 0, 0) != 0) die("cvec clear failed");
    double h_norm = 0.0;
    {
        int64_t nr = 0;
        cap.enabled = true;
        for (int i = 0; i < NP; ++i) {
            llama_memory_clear(mem, true);
            for (int cur = 0; cur < plen; cur += ubatch) {
                const int n = std::min(ubatch, plen - cur);
                batch.n_tokens = n;
                for (int j = 0; j < n; ++j) {
                    batch.token[j] = pre[i].toks[cur + j]; batch.pos[j] = cur + j; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
                    batch.logits[j] = 0;
                }
                cap.rows = 0;
                if (llama_decode(ctx, batch) != 0) die("llama_decode failed (norm pass)");
                for (int j = 16 - cur > 0 ? 16 - cur : 0; j < n; ++j) {
                    double s2 = 0.0;
                    const float * h = cap.buf.data() + (size_t) j * n_embd;
                    for (int e = 0; e < n_embd; ++e) s2 += (double) h[e] * h[e];
                    h_norm += std::sqrt(s2);
                    nr++;
                }
            }
        }
        cap.enabled = false;
        h_norm /= std::max<int64_t>(1, nr);
    }

    // the unsteered prefixes [0, plen - 1), decoded once into the master sequences NP + i
    llama_memory_clear(mem, true);
    {
        int nb = 0;
        for (int i = 0; i < NP; ++i) {
            for (int t = 0; t + 1 < plen; ++t) {
                batch.token[nb] = pre[i].toks[t]; batch.pos[nb] = t; batch.n_seq_id[nb] = 1; batch.seq_id[nb][0] = NP + i;
                batch.logits[nb] = 0;
                if (++nb == ubatch) {
                    batch.n_tokens = nb;
                    if (llama_decode(ctx, batch) != 0) die("llama_decode failed (prefix)");
                    nb = 0;
                }
            }
        }
        if (nb > 0) {
            batch.n_tokens = nb;
            if (llama_decode(ctx, batch) != 0) die("llama_decode failed (prefix)");
        }
    }

    // common random numbers: one uniform per (prefix, step)
    std::vector<std::vector<double>> Uni(NP, std::vector<double>(G));
    for (int i = 0; i < NP; ++i) {
        std::mt19937_64 rng((uint64_t) seed * 1000003ull + (uint64_t) i * 104729ull);
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        for (auto & u : Uni[i]) u = ud(rng);
    }

    std::vector<float> cvec((size_t) n_embd * n_layer, 0.0f);
    std::vector<std::vector<float>> lp(NP, std::vector<float>(n_vocab));

    // one arm: steer with sc * u after block `layer` (u null: no steering)
    auto run_arm = [&](const std::vector<float> * u, float sc) {
        if (u == nullptr) {
            if (llama_set_adapter_cvec(ctx, nullptr, 0, n_embd, 0, 0) != 0) die("cvec clear failed");
        } else {
            std::fill(cvec.begin(), cvec.end(), 0.0f);
            for (int e = 0; e < n_embd; ++e) cvec[(size_t) (layer - 1) * n_embd + e] = sc * (*u)[e];
            if (llama_set_adapter_cvec(ctx, cvec.data(), cvec.size(), n_embd, layer, layer) != 0) die("cvec set failed");
        }
        arm_out o;
        o.h.assign((size_t) NP * G * n_embd, NAN);
        o.ent.assign((size_t) NP * G, NAN);
        o.tok.assign((size_t) NP * G, -1);
        o.endpoint.assign(NP, 0);

        for (int i = 0; i < NP; ++i) {
            llama_memory_seq_rm(mem, i, -1, -1);
            llama_memory_seq_cp(mem, NP + i, i, -1, -1);
        }
        std::vector<int> act(NP);
        for (int i = 0; i < NP; ++i) {
            act[i] = i;
            batch.token[i] = pre[i].toks[plen - 1]; batch.pos[i] = plen - 1; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = i;
            batch.logits[i] = 1;
        }
        batch.n_tokens = NP;
        std::vector<std::vector<int32_t>> gen(NP);
        std::vector<int> hot(NP, 0);
        std::vector<const float *> zr(NP);
        for (int s = 0; s < G && !act.empty(); ++s) {
            // decode the current token of every active sequence: its layer-L state is lag s
            cap.enabled = true;
            cap.rows = 0;
            if (llama_decode(ctx, batch) != 0) die("llama_decode failed (gen)");
            cap.enabled = false;
            const int na = (int) act.size();
            if (cap.rows != na) die("captured row count != active sequences");
            for (int j = 0; j < na; ++j) {
                const int i = act[j];
                float * dst = o.h.data() + ((size_t) i * G + s) * n_embd;
                const float * src = cap.buf.data() + (size_t) j * n_embd;
                for (int e = 0; e < n_embd; ++e) dst[e] = src[e] - (u ? sc * (*u)[e] : 0.0f); // pre-steer state
                zr[j] = llama_get_logits_ith(ctx, j);
            }
            parallel_for(na, std::min(na, n_threads), [&](int j) {
                const int i = act[j];
                float * l = lp[j].data();
                log_softmax(zr[j], l, n_vocab);
                double H = 0.0, tot = 0.0;
                for (int v = 0; v < n_vocab; ++v) {
                    const double p = std::exp((double) l[v]);
                    H -= p * l[v];
                    tot += p;
                }
                const double target = Uni[i][s] * tot;
                double c = 0.0;
                int tok = n_vocab - 1;
                for (int v = 0; v < n_vocab; ++v) {
                    c += std::exp((double) l[v]);
                    if (c >= target) { tok = v; break; }
                }
                gen[i].push_back(tok);
                o.tok[(size_t) i * G + s] = tok;
                o.ent[(size_t) i * G + s] = (float) H;
            });
            o.n_tok += na;
            std::vector<int> nxt;
            for (int j = 0; j < na; ++j) {
                const int i = act[j];
                hot[i] = o.ent[(size_t) i * G + s] > incoh_h ? hot[i] + 1 : 0;
                if (hot[i] >= incoh_run) { o.endpoint[i] = 2; continue; }
                if (find_loop(gen[i], loop_max_p, loop_min_span) > 0) { o.endpoint[i] = 1; continue; }
                nxt.push_back(i);
            }
            act = nxt;
            for (int k = 0; k < (int) act.size(); ++k) {
                const int i = act[k];
                batch.token[k] = gen[i].back(); batch.pos[k] = plen + s; batch.n_seq_id[k] = 1; batch.seq_id[k][0] = i;
                batch.logits[k] = 1;
            }
            batch.n_tokens = (int) act.size();
        }
        return o;
    };

    // ---- record layout ----
    const size_t n_cell = (size_t) NP * G;
    const size_t rec_bytes = n_cell * K * sizeof(ggml_fp16_t) + n_cell * sizeof(float) * 2 + n_cell * sizeof(int32_t) + (size_t) NP * sizeof(int32_t) * 2;
    const std::string bin_path = out_prefix + ".bin";
    const long long have = file_size(bin_path);
    long long done = have > 0 ? have / (long long) rec_bytes : 0;
    if (have > 0 && have % (long long) rec_bytes != 0) {
        fprintf(stderr, "homeo-ident: %s has a partial record; it is ignored and overwritten\n", bin_path.c_str());
    }
    const long long n_rec = 1 + (long long) doses.size() * n_probe * 2;

    {
        json meta;
        meta["model"] = model_path; meta["rollouts"] = rollouts_path; meta["basis"] = basis_path; meta["k"] = K;
        meta["layer"] = layer; meta["resid_norm"] = h_norm; meta["n_prefix"] = NP; meta["prefix_len"] = plen; meta["n_gen"] = G;
        meta["seed"] = seed; meta["probe_seed"] = probe_seed; meta["n_probe"] = n_probe; meta["doses"] = doses;
        meta["record_order"] = "unsteered, then for dose in doses, probe in 0..n_probe-1, sign in (+, -)";
        meta["record_layout"] = "f16 dproj [NP][G][K], f32 dnorm [NP][G], f32 ent [NP][G], i32 tok [NP][G], i32 first_div [NP], i32 endpoint [NP]";
        meta["record_bytes"] = rec_bytes;
        meta["lag0"] = "state of the last prefix token (no generated token yet): its response must be 0";
        meta["endpoints"] = { { "loop_max_period", loop_max_p }, { "loop_min_span", loop_min_span }, { "incoh_h", incoh_h }, { "incoh_run", incoh_run } };
        json docs = json::array();
        for (auto & p : pre) docs.push_back({ p.doc, p.rollout });
        meta["prefixes"] = docs;
        std::ofstream(out_prefix + ".meta.json") << meta.dump(2) << "\n";
        FILE * fz = fopen((out_prefix + ".z.f32").c_str(), "wb");
        if (!fz || fwrite(Z.data(), sizeof(float), Z.size(), fz) != Z.size()) die("cannot write probe coordinates");
        fclose(fz);
    }
    fprintf(stderr, "homeo-ident: layer %d, mean |h| %.1f, %d prefixes x %d tokens, %d generated, K %d, %d probes x %zu doses; %lld of %lld records done\n",
            layer, h_norm, NP, plen, G, K, n_probe, doses.size(), done, n_rec);

    FILE * fb = fopen(bin_path.c_str(), have > 0 ? "r+b" : "wb");
    if (!fb) die("cannot open output .bin");
    fseek(fb, (long) (done * (long long) rec_bytes), SEEK_SET);

    FILE * flog = nullptr;
    if (!log_path.empty()) {
        bool fresh = true;
        if (FILE * t = fopen(log_path.c_str(), "rb")) { fseek(t, 0, SEEK_END); fresh = ftell(t) == 0; fclose(t); }
        flog = fopen(log_path.c_str(), "ab");
        if (!flog) die("cannot open --log");
        if (fresh) fprintf(flog, "date,experiment,model,manipulation,dose_frac_resid,tokens_in_perturbed_state,n_contexts,endpoint,frac_endpoint_hit,notes\n");
    }
    char endpoint_desc[160];
    snprintf(endpoint_desc, sizeof(endpoint_desc), "loop: period<=%d over max(3p;%d) tokens | incoh: H>%.1f for %d steps", loop_max_p, loop_min_span, incoh_h, incoh_run);

    // the unsteered arm is always recomputed (needed as the reference); it is deterministic
    const auto t0 = std::chrono::steady_clock::now();
    arm_out base = run_arm(nullptr, 0.0f);
    std::vector<ggml_fp16_t> dproj(n_cell * K);
    std::vector<float> dnorm(n_cell);
    std::vector<int32_t> first_div(NP);
    auto write_rec = [&](const arm_out & o) {
        parallel_for(NP, std::min(NP, n_threads), [&](int i) {
            first_div[i] = -1;
            for (int s = 0; s < G; ++s) {
                const size_t c = (size_t) i * G + s;
                if (first_div[i] < 0 && o.tok[c] != base.tok[c]) first_div[i] = s;
                const float * h = o.h.data() + c * n_embd, * h0 = base.h.data() + c * n_embd;
                if (std::isnan(h[0]) || std::isnan(h0[0])) {
                    for (int k = 0; k < K; ++k) dproj[c * K + k] = ggml_fp32_to_fp16(NAN);
                    dnorm[c] = NAN;
                    continue;
                }
                double n2 = 0.0;
                for (int e = 0; e < n_embd; ++e) { const double d = (double) h[e] - h0[e]; n2 += d * d; }
                dnorm[c] = (float) std::sqrt(n2);
                for (int k = 0; k < K; ++k) {
                    const float * q = Q.data() + (size_t) k * n_embd;
                    double d = 0.0;
                    for (int e = 0; e < n_embd; ++e) d += (double) q[e] * ((double) h[e] - h0[e]);
                    dproj[c * K + k] = ggml_fp32_to_fp16((float) d);
                }
            }
        });
        fwrite(dproj.data(), sizeof(ggml_fp16_t), dproj.size(), fb);
        fwrite(dnorm.data(), sizeof(float), dnorm.size(), fb);
        fwrite(o.ent.data(), sizeof(float), o.ent.size(), fb);
        fwrite(o.tok.data(), sizeof(int32_t), o.tok.size(), fb);
        fwrite(first_div.data(), sizeof(int32_t), NP, fb);
        fwrite(o.endpoint.data(), sizeof(int32_t), NP, fb);
        fflush(fb);
    };
    if (done == 0) {
        write_rec(base);
        done = 1;
    }
    {
        // lag-0 check on the unsteered arm against itself is trivial; report its endpoints instead
        int ne = 0;
        for (int i = 0; i < NP; ++i) ne += base.endpoint[i] != 0;
        fprintf(stderr, "homeo-ident: unsteered arm done (%d endpoints of %d)\n", ne, NP);
    }

    char date[16];
    { const time_t now = time(nullptr); strftime(date, sizeof(date), "%Y-%m-%d", localtime(&now)); }
    int64_t blk_tok = 0;
    int blk_arms = 0, blk_ctx_end = 0, blk_ctx = 0;
    float blk_dose = 0.0f;
    auto log_block = [&]() {
        if (!flog || blk_arms == 0) return;
        char manip[256];
        snprintf(manip, sizeof(manip), "steer after block %d along +-random unit directions in a %d-dim PCA subspace x dose x mean|h| %.1f (%d arms)",
                 layer, K, h_norm, blk_arms);
        std::string notes = "homeo-ident (plan B); steering from the last prefix token on (prefix unsteered)";
        if (!note.empty()) notes += "; " + note;
        fprintf(flog, "%s,B-ident,%s,%s,%.3f,%lld,%d,%s,%.3f,%s\n", date, csv_q(base_name(model_path)).c_str(), csv_q(manip).c_str(),
                blk_dose, (long long) blk_tok, blk_ctx, csv_q(endpoint_desc).c_str(), blk_ctx ? (double) blk_ctx_end / blk_ctx : 0.0, csv_q(notes).c_str());
        fflush(flog);
        blk_tok = 0; blk_arms = 0; blk_ctx_end = 0; blk_ctx = 0;
    };

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    long long rec = 1;
    for (size_t di = 0; di < doses.size() && !g_stop; ++di) {
        const float sc_abs = (float) (doses[di] * h_norm);
        for (int p = 0; p < n_probe && !g_stop; ++p) {
            for (int sg = 0; sg < 2 && !g_stop; ++sg, ++rec) {
                if (rec < done) continue;
                {
                    std::ifstream pf(out_prefix + ".pause_ms");
                    int v;
                    if (pf >> v && v >= 0) pause_ms = v;
                }
                const auto ta = std::chrono::steady_clock::now();
                arm_out o = run_arm(&U[p], sg == 0 ? sc_abs : -sc_abs);
                write_rec(o);
                int ne = 0, nd = 0;
                for (int i = 0; i < NP; ++i) { ne += o.endpoint[i] != 0; nd += first_div[i] >= 0; }
                if (blk_arms > 0 && blk_dose != doses[di]) log_block();
                blk_dose = doses[di]; blk_tok += o.n_tok; blk_arms++; blk_ctx += NP; blk_ctx_end += ne;
                if (blk_arms >= log_every) log_block();
                const double sa = std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count();
                const double st = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                fprintf(stderr, "  dose %.3f probe %4d %c: diverged %2d/%d, endpoints %d, %.1f s (record %lld/%lld, %.2f h)\n",
                        doses[di], p, sg == 0 ? '+' : '-', nd, NP, ne, sa, rec + 1, n_rec, st / 3600.0);
                for (int w = 0; w < pause_ms && !g_stop; w += 100) std::this_thread::sleep_for(std::chrono::milliseconds(std::min(100, pause_ms - w)));
            }
        }
    }
    log_block();
    if (g_stop) fprintf(stderr, "homeo-ident: stopped by signal after record %lld; rerun to resume\n", rec);
    fclose(fb);
    if (flog) fclose(flog);
    fprintf(stderr, "homeo-ident: done -> %s\n", bin_path.c_str());
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
