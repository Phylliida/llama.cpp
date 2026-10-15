// homeo-loop: loop margin under steering (K4 in NOTES-homeostasis.md).
//
// For each direction and dose (in units of the mean residual norm at --layer), the unit direction
// times dose * norm is added after block --layer at every position (prefix and generated). Then
// --n-gen tokens are sampled from --n-prefix held-out prefixes (doc_idx % 4 == 0) at temp 1 with
// no truncation. Each prefix has its own uniforms, shared by all directions and doses (common
// random numbers, inverse-CDF sampling).
// Humane endpoints (Three Rs protocol): a sequence stops on
//   loop:  the last max(3p, --loop-min-span) tokens repeat with period p <= --loop-max-period
//   incoh: entropy > --incoh-h nats for --incoh-run steps in a row
// Ladder: per direction the doses go up, and stop after the first dose where >= --stop-frac of
// the sequences hit the loop endpoint, or (with --stop-any K) after K doses in a row where >= --stop-frac
// hit any endpoint (loop or incoherence): past that point more dose answers nothing new.
// Stopped sequences leave the batch, so the batch shape differs between arms (~2% residual
// numerics, see NOTES). This is fine for a coarse readout like loop onset.
//
// usage: homeo-loop -m model.gguf --rollouts rollouts.jsonl --layer L --out k4.jsonl
//          [--vecs a.f32,b.f32] [--neg] [--random K] [--doses 0.25,0.5,0.75,1,1.5,2,3]
//          [--n-prefix 16] [--prefix-len 256] [--n-gen 128] [--seed 1] [--ngl 999] [--threads N]
//          [--loop-max-period 16] [--loop-min-span 12] [--incoh-h 6] [--incoh-run 16] [--stop-frac 0.75]
//          [--log welfare_log.csv] [--note text]
// vec files are raw f32 [n_embd], unit-normalized on load; --neg also runs each file vector negated.
// --layer-end E adds the same dose * norm * u after every block in [--layer, E] (multi-layer steering
// over a band, like hand-made control vectors); the norm is still measured at --layer.
// Dose 0 (no steering) runs once. Output: one JSON line per (direction, dose, sequence) in --out,
// run info in --out.meta.json, and one row per steered (direction, dose) appended to --log.

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
#include <random>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

static void die(const char * msg) {
    fprintf(stderr, "homeo-loop: %s\n", msg);
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

static void log_softmax(const float * z, float * out, int n) {
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) m = std::max(m, z[i]);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp(z[i] - m);
    const float lse = m + (float) std::log(s);
    for (int i = 0; i < n; ++i) out[i] = z[i] - lse;
}

// smallest period p whose repeat covers the last max(3p, min_span) tokens; 0 if none.
// *start gets the first index of the periodic run.
static int find_loop(const std::vector<int32_t> & g, int max_p, int min_span, int * start) {
    const int n = (int) g.size();
    for (int p = 1; p <= max_p; ++p) {
        const int span = std::max(3 * p, min_span);
        if (n < span) break;
        bool ok = true;
        for (int i = n - span + p; i < n && ok; ++i) ok = g[i] == g[i - p];
        if (ok) {
            int k = n - span;
            while (k > 0 && g[k - 1] == g[k - 1 + p]) k--;
            *start = k;
            return p;
        }
    }
    return 0;
}

// ---- l_out capture for the residual norm ----

struct capture {
    std::string          name;
    std::vector<float>   buf;
    int64_t              rows    = 0;
    int64_t              n_embd  = 0;
    bool                 enabled = false;
};

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    auto * c = (capture *) ud;
    if (!c->enabled || c->name != t->name) {
        return !ask;
    }
    if (ask) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd || !ggml_is_contiguous(t)) {
        die("unexpected l_out tensor layout");
    }
    c->rows = ggml_nrows(t);
    c->buf.resize(c->rows * c->n_embd);
    ggml_backend_tensor_get(t, c->buf.data(), 0, ggml_nbytes(t));
    return true;
}

struct direction {
    std::string        name;
    std::vector<float> u; // unit norm
};

struct seq_out {
    std::vector<int32_t> toks;
    std::vector<float>   ent;
    int loop_onset = -1, loop_period = 0, loop_detect = -1, incoh_onset = -1, hot = 0;
};

static std::string csv_q(const std::string & s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"') o += '"'; o += c; }
    return o + "\"";
}

int main(int argc, char ** argv) {
    std::string model_path, rollouts_path, out_path, vecs_str, log_path, note;
    std::string doses_str = "0.25,0.5,0.75,1,1.5,2,3";
    int layer = -1, layer_end = -1, n_prefix = 16, plen = 256, n_gen = 128, seed = 1, ngl = 999, n_random = 0;
    int loop_max_p = 16, loop_min_span = 12, incoh_run = 16, ubatch = 512;
    float incoh_h = 6.0f, stop_frac = 0.75f;
    int stop_any = 0;
    bool neg = false;
    int n_threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) die(("missing value for " + a).c_str()); return argv[++i]; };
        if      (a == "-m")                model_path    = next();
        else if (a == "--rollouts")        rollouts_path = next();
        else if (a == "--out")             out_path      = next();
        else if (a == "--layer")           layer         = atoi(next());
        else if (a == "--layer-end")       layer_end     = atoi(next());
        else if (a == "--vecs")            vecs_str      = next();
        else if (a == "--neg")             neg           = true;
        else if (a == "--random")          n_random      = atoi(next());
        else if (a == "--doses")           doses_str     = next();
        else if (a == "--n-prefix")        n_prefix      = atoi(next());
        else if (a == "--prefix-len")      plen          = atoi(next());
        else if (a == "--n-gen")           n_gen         = atoi(next());
        else if (a == "--seed")            seed          = atoi(next());
        else if (a == "--ngl")             ngl           = atoi(next());
        else if (a == "--threads")         n_threads     = atoi(next());
        else if (a == "--loop-max-period") loop_max_p    = atoi(next());
        else if (a == "--loop-min-span")   loop_min_span = atoi(next());
        else if (a == "--incoh-h")         incoh_h       = (float) atof(next());
        else if (a == "--incoh-run")       incoh_run     = atoi(next());
        else if (a == "--stop-frac")       stop_frac     = (float) atof(next());
        else if (a == "--stop-any")        stop_any      = atoi(next());
        else if (a == "--log")             log_path      = next();
        else if (a == "--note")            note          = next();
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || rollouts_path.empty() || out_path.empty() || layer < 1) {
        fprintf(stderr, "usage: homeo-loop -m model.gguf --rollouts rollouts.jsonl --layer L (>= 1) --out k4.jsonl [options]\n");
        return 1;
    }
    std::vector<float> doses;
    for (auto & s : split(doses_str)) doses.push_back((float) atof(s.c_str()));
    std::sort(doses.begin(), doses.end());
    doses.erase(std::remove_if(doses.begin(), doses.end(), [](float d) { return d <= 0.0f; }), doses.end());

    // ---- held-out prefixes ----
    struct prefix { std::vector<int32_t> toks; int doc; int rollout; };
    std::vector<prefix> pre;
    struct ro { std::vector<int32_t> prompt; std::string text; int doc; int rollout; };
    std::vector<ro> rolls;
    {
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
    if (layer >= n_layer - 1) die("--layer must be < n_layer - 1 (last layer output is pruned to output rows)");
    if (layer_end < layer) layer_end = layer;
    if (layer_end >= n_layer - 1) die("--layer-end must be < n_layer - 1");

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
    if (NP < 1) die("no held-out prefixes long enough");
    if (NP < n_prefix) fprintf(stderr, "homeo-loop: only %d held-out prefixes of %d tokens\n", NP, plen);

    // ---- directions ----
    std::vector<direction> dirs;
    auto add_dir = [&](const std::string & name, std::vector<float> v) {
        double s2 = 0.0;
        for (float x : v) s2 += (double) x * x;
        if (s2 <= 0.0) die(("zero vector: " + name).c_str());
        const float inv = (float) (1.0 / std::sqrt(s2));
        for (float & x : v) x *= inv;
        dirs.push_back({ name, std::move(v) });
    };
    for (auto & path : split(vecs_str)) {
        std::vector<float> v(n_embd);
        FILE * f = fopen(path.c_str(), "rb");
        if (!f || fread(v.data(), sizeof(float), n_embd, f) != (size_t) n_embd) die(("cannot read vec (raw f32 [n_embd]): " + path).c_str());
        fclose(f);
        const std::string name = base_name(path);
        if (neg) {
            std::vector<float> vn(v);
            for (float & x : vn) x = -x;
            add_dir(name, v);
            add_dir(name + "_neg", vn);
        } else {
            add_dir(name, v);
        }
    }
    for (int k = 0; k < n_random; ++k) {
        std::mt19937_64 rng((uint64_t) seed * 1000003ull + (uint64_t) layer * 7919ull + k);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> v(n_embd);
        for (float & x : v) x = nd(rng);
        add_dir("rand" + std::to_string(k), v);
    }

    // ---- context ----
    capture cap;
    cap.name   = "l_out-" + std::to_string(layer);
    cap.n_embd = n_embd;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = NP * (plen + n_gen) + 256;
    cp.n_batch           = ubatch;
    cp.n_ubatch          = ubatch;
    cp.n_seq_max         = NP;
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
    llama_batch batch = llama_batch_init(std::max(ubatch, NP), 0, 1);

    // ---- mean residual norm at the layer (no steering), one prefix per call so rows are in order ----
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
                if (cap.rows != n) die("captured row count != batch size");
                for (int j = 0; j < n; ++j) {
                    if (cur + j < 16) continue; // skip BOS / sink positions
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
    fprintf(stderr, "homeo-loop: layer %d, mean |h| %.1f over %d prefixes of %d tokens, %zu directions\n", layer, h_norm, NP, plen, dirs.size());

    // common random numbers: one uniform per (prefix, step)
    std::vector<std::vector<double>> U(NP, std::vector<double>(n_gen));
    for (int i = 0; i < NP; ++i) {
        std::mt19937_64 rng((uint64_t) seed * 1000003ull + (uint64_t) i * 104729ull);
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        for (auto & u : U[i]) u = ud(rng);
    }

    std::vector<float> cvec((size_t) n_embd * n_layer, 0.0f);
    std::vector<std::vector<float>> lp(NP, std::vector<float>(n_vocab));

    // one arm: steer with dose * h_norm * u (dose 0: no steering), returns the sequences
    auto run_arm = [&](const direction * d, float dose) {
        if (d == nullptr || dose == 0.0f) {
            if (llama_set_adapter_cvec(ctx, nullptr, 0, n_embd, 0, 0) != 0) die("cvec clear failed");
        } else {
            const float sc = (float) (dose * h_norm);
            for (int il = layer; il <= layer_end; ++il) {
                for (int e = 0; e < n_embd; ++e) cvec[(size_t) (il - 1) * n_embd + e] = sc * d->u[e];
            }
            if (llama_set_adapter_cvec(ctx, cvec.data(), cvec.size(), n_embd, layer, layer_end) != 0) die("cvec set failed");
        }
        // prefill tokens [0, plen - 1) of every prefix, packed, no logits
        llama_memory_clear(mem, true);
        int nb = 0;
        auto flush = [&]() {
            if (nb == 0) return;
            batch.n_tokens = nb;
            if (llama_decode(ctx, batch) != 0) die("llama_decode failed (prefill)");
            nb = 0;
        };
        for (int i = 0; i < NP; ++i) {
            for (int t = 0; t + 1 < plen; ++t) {
                batch.token[nb] = pre[i].toks[t]; batch.pos[nb] = t; batch.n_seq_id[nb] = 1; batch.seq_id[nb][0] = i;
                batch.logits[nb] = 0;
                if (++nb == ubatch) flush();
            }
        }
        flush();
        // the last prefix token of every sequence in one batch: logits for the first sample
        std::vector<int> act(NP);
        for (int i = 0; i < NP; ++i) {
            act[i] = i;
            batch.token[i] = pre[i].toks[plen - 1]; batch.pos[i] = plen - 1; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = i;
            batch.logits[i] = 1;
        }
        batch.n_tokens = NP;
        if (llama_decode(ctx, batch) != 0) die("llama_decode failed (prefix end)");

        std::vector<seq_out> out(NP);
        std::vector<const float *> zr(NP);
        for (int s = 0; s < n_gen && !act.empty(); ++s) {
            const int na = (int) act.size();
            for (int j = 0; j < na; ++j) zr[j] = llama_get_logits_ith(ctx, j);
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
                // inverse CDF with the shared uniform
                const double target = U[i][s] * tot;
                double c = 0.0;
                int tok = n_vocab - 1;
                for (int v = 0; v < n_vocab; ++v) {
                    c += std::exp((double) l[v]);
                    if (c >= target) { tok = v; break; }
                }
                out[i].toks.push_back(tok);
                out[i].ent.push_back((float) H);
            });
            // endpoints
            std::vector<int> next;
            for (int j = 0; j < na; ++j) {
                const int i = act[j];
                seq_out & o = out[i];
                o.hot = o.ent.back() > incoh_h ? o.hot + 1 : 0;
                if (o.hot >= incoh_run) {
                    o.incoh_onset = s - incoh_run + 1;
                    continue;
                }
                int start = -1;
                const int p = find_loop(o.toks, loop_max_p, loop_min_span, &start);
                if (p > 0) {
                    o.loop_period = p; o.loop_onset = start; o.loop_detect = s;
                    continue;
                }
                next.push_back(i);
            }
            if (s + 1 == n_gen || next.empty()) break;
            for (int k = 0; k < (int) next.size(); ++k) {
                const int i = next[k];
                batch.token[k] = out[i].toks.back(); batch.pos[k] = plen + s; batch.n_seq_id[k] = 1; batch.seq_id[k][0] = i;
                batch.logits[k] = 1;
            }
            batch.n_tokens = (int) next.size();
            if (llama_decode(ctx, batch) != 0) die("llama_decode failed (gen)");
            act = next;
        }
        return out;
    };

    std::ofstream fo(out_path);
    if (!fo) die("cannot open --out");
    FILE * flog = nullptr;
    if (!log_path.empty()) {
        bool fresh = true;
        if (FILE * t = fopen(log_path.c_str(), "rb")) { fseek(t, 0, SEEK_END); fresh = ftell(t) == 0; fclose(t); }
        flog = fopen(log_path.c_str(), "ab");
        if (!flog) die("cannot open --log");
        if (fresh) fprintf(flog, "date,experiment,model,manipulation,dose_frac_resid,tokens_in_perturbed_state,n_contexts,endpoint,frac_endpoint_hit,notes\n");
    }
    char date[16];
    { const time_t now = time(nullptr); strftime(date, sizeof(date), "%Y-%m-%d", localtime(&now)); }
    char endpoint_desc[160];
    snprintf(endpoint_desc, sizeof(endpoint_desc), "loop: period<=%d over max(3p;%d) tokens | incoh: H>%.1f for %d steps",
             loop_max_p, loop_min_span, incoh_h, incoh_run);

    auto piece = [&](int32_t t) {
        char buf[256];
        const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        return n > 0 ? std::string(buf, n) : std::string();
    };

    // write one arm, return (loop fraction, any-endpoint fraction)
    auto emit = [&](const std::string & name, float dose, const std::vector<seq_out> & out, double secs) {
        int n_loop = 0, n_inc = 0;
        int64_t n_tok = 0;
        for (int i = 0; i < NP; ++i) {
            const seq_out & o = out[i];
            n_loop += o.loop_period > 0;
            n_inc  += o.incoh_onset >= 0;
            n_tok  += (int64_t) o.toks.size();
            std::string text;
            for (int32_t t : o.toks) text += piece(t);
            json j;
            j["layer"] = layer; j["layer_end"] = layer_end; j["vector"] = name; j["dose"] = dose; j["added_norm"] = dose * h_norm;
            j["seq"] = i; j["doc"] = pre[i].doc; j["rollout"] = pre[i].rollout;
            j["steps_run"] = o.toks.size();
            j["loop_onset"] = o.loop_onset; j["loop_period"] = o.loop_period; j["loop_detect"] = o.loop_detect;
            j["incoh_onset"] = o.incoh_onset;
            j["tokens"] = o.toks; j["entropies"] = o.ent;
            j["text"] = text; // invalid UTF-8 is replaced at dump time
            fo << j.dump(-1, ' ', false, json::error_handler_t::replace) << "\n";
        }
        fo.flush();
        const double f_loop = (double) n_loop / NP, f_any = (double) (n_loop + n_inc) / NP;
        double h0 = 0.0; int nh = 0;
        for (auto & o : out) for (int s = 0; s < (int) o.ent.size() && s < 32; ++s) { h0 += o.ent[s]; nh++; }
        fprintf(stderr, "  %-24s dose %4.2f (|add| %7.1f): loop %5.2f  incoh %5.2f  mean H(first 32) %5.2f  tokens %6lld  %.1f s\n",
                name.c_str(), dose, dose * h_norm, f_loop, (double) n_inc / NP, h0 / std::max(1, nh), (long long) n_tok, secs);
        if (flog && dose > 0.0f) {
            char manip[256];
            if (layer_end > layer) {
                snprintf(manip, sizeof(manip), "steer after blocks %d-%d along %s (unit) x dose x mean|h_%d| %.1f", layer, layer_end, name.c_str(), layer, h_norm);
            } else {
                snprintf(manip, sizeof(manip), "steer after block %d along %s (unit) x dose x mean|h| %.1f", layer, name.c_str(), h_norm);
            }
            std::string notes = "prefill also steered: " + std::to_string(NP) + " x " + std::to_string(plen) + " tokens";
            if (!note.empty()) notes += "; " + note;
            fprintf(flog, "%s,K4,%s,%s,%.3f,%lld,%d,%s,%.3f,%s\n", date, csv_q(base_name(model_path)).c_str(), csv_q(manip).c_str(),
                    dose, (long long) n_tok, NP, csv_q(endpoint_desc).c_str(), f_any, csv_q(notes).c_str());
            fflush(flog);
        }
        return std::make_pair(f_loop, f_any);
    };

    const auto t0 = std::chrono::steady_clock::now();
    auto since = [&](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); };

    json meta;
    meta["model"] = model_path; meta["rollouts"] = rollouts_path; meta["layer"] = layer; meta["layer_end"] = layer_end; meta["resid_norm"] = h_norm;
    meta["n_prefix"] = NP; meta["prefix_len"] = plen; meta["n_gen"] = n_gen; meta["seed"] = seed;
    meta["doses"] = doses; meta["stop_frac"] = stop_frac; meta["stop_any"] = stop_any; meta["endpoints"] = endpoint_desc;
    meta["vectors"] = json::array();
    for (auto & d : dirs) meta["vectors"].push_back(d.name);
    {
        // cosines between directions (unit vectors)
        json cs = json::array();
        for (size_t a = 0; a < dirs.size(); ++a) {
            std::vector<double> row;
            for (size_t b = 0; b < dirs.size(); ++b) {
                double c = 0.0;
                for (int e = 0; e < n_embd; ++e) c += (double) dirs[a].u[e] * dirs[b].u[e];
                row.push_back(c);
            }
            cs.push_back(row);
        }
        meta["cosines"] = cs;
    }

    {
        const auto t = std::chrono::steady_clock::now();
        auto out = run_arm(nullptr, 0.0f);
        emit("none", 0.0f, out, since(t));
    }
    json ladders = json::object();
    for (auto & d : dirs) {
        json lad = json::array();
        int run_any = 0;
        for (float dose : doses) {
            const auto t = std::chrono::steady_clock::now();
            auto out = run_arm(&d, dose);
            const auto fr = emit(d.name, dose, out, since(t));
            lad.push_back({ { "dose", dose }, { "loop_frac", fr.first }, { "endpoint_frac", fr.second } });
            if (fr.first >= stop_frac) break; // answer, then stop
            run_any = fr.second >= stop_frac ? run_any + 1 : 0;
            if (stop_any > 0 && run_any >= stop_any) break;
        }
        ladders[d.name] = lad;
    }
    meta["ladders"] = ladders;
    meta["seconds"] = since(t0);
    std::ofstream(out_path + ".meta.json") << meta.dump(2) << "\n";
    fprintf(stderr, "homeo-loop: done in %.0f s -> %s\n", since(t0), out_path.c_str());

    if (flog) fclose(flog);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
