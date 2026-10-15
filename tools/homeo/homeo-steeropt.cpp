// homeo-steeropt: optimize one steering vector through the model (S2(c) in NOTES-homeostasis.md).
//
// Target per position: the unsteered distribution p with band tokens upweighted by --upweight U,
//   r ~ p * (1 + (U - 1) [x in band])
// i.e. "promote only band tokens, keep everything else in proportion". Band: p in [P/w, P*w]
// (--band P --width w) or raw logit in [Z-d, Z+d] (--logit-band Z --logit-width d).
// v [n_embd] is added at every position after block --layer; loss = mean_t H(r_t, q_t), backward
// through the jspace GLM-4 graph, Adam on the host. Train and test windows come from disjoint
// documents (doc_idx % 4 == 0 -> test). On test windows it reports CE for v = 0, for the best
// global temperature, and for v, plus the band-mass ratio and the KL between the renormalized
// non-band parts of p and q (the "only" part: 0 means nothing outside the band moved).
//
// usage: homeo-steeropt -m model.gguf --rollouts rollouts.jsonl --layer L (--band P | --logit-band Z)
//          --out v.f32 [--upweight 4] [--width 1.5] [--logit-width 1] [--n-tokens 128] [--n-train 48]
//          [--n-test 16] [--steps 200] [--accum 2] [--lr 0.03] [--skip 16] [--eval-every 50] [--seed 0]
//          [--fd-check N]   compare grad.d with central differences along N random directions, then exit
//          [--target-temp T] positive control: target r ~ p^(1/T) instead of a band (known to be reachable)
//          [--layers a,b,...] sweep: optimize one vector per layer in this process; --out is then a prefix
//                            (PREFIX_LNN.f32 per layer, PREFIX.json for all)
//          [--lr-frac f]    per-layer Adam lr = f * mean |h_l| / sqrt(n_embd) (default: fixed --lr)
//          [--fixed-norm s] search directions at |v| = s * mean |h_l|: start on that sphere (random direction,
//                           or --init-vec's direction), project gradients onto the tangent space, renormalize
//                           after each step. Tests effects that only appear at large doses.
//          [--init-vec v.f32] starting direction for --fixed-norm

#include "jlens.h"
#include "llama.h"

#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

static void die(const char * msg) {
    fprintf(stderr, "homeo-steeropt: %s\n", msg);
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

static float log_softmax(const float * z, float * out, int n) {
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) m = std::max(m, z[i]);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp(z[i] - m);
    const float lse = m + (float) std::log(s);
    for (int i = 0; i < n; ++i) out[i] = z[i] - lse;
    return lse;
}

struct window {
    std::vector<int32_t> toks;
    std::vector<float>   lp;    // unsteered log p [n_tokens * V]
    std::vector<float>   lse;   // unsteered logsumexp per row (raw logit = lp + lse)
    std::vector<float>   r;     // target [n_tokens * V]
    std::vector<uint8_t> has_band; // per row: band non-empty and row >= skip
};

struct row_metrics {
    double ce = 0, band_ratio = 0, kl_nonband = 0, top_changed = 0;
    int    n_band = 0, n = 0;
};

int main(int argc, char ** argv) {
    std::string model_path, rollouts_path, out_path;
    int   layer = -1, n_tokens = 128, n_train = 48, n_test = 16, steps = 200, accum = 2, skip = 16, eval_every = 50, seed = 0;
    float band_p = -1.0f, width = 1.5f, logit_band = NAN, logit_width = 1.0f, upweight = 4.0f, lr = 0.03f;
    int   n_threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    int   fd_check = 0;
    float target_temp = 0.0f, lr_frac = 0.0f, fixed_norm = 0.0f;
    std::string layers_str, init_vec_path;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) die(("missing value for " + a).c_str()); return argv[++i]; };
        if      (a == "-m")            model_path    = next();
        else if (a == "--rollouts")    rollouts_path = next();
        else if (a == "--out")         out_path      = next();
        else if (a == "--layer")       layer         = atoi(next());
        else if (a == "--band")        band_p        = (float) atof(next());
        else if (a == "--width")       width         = (float) atof(next());
        else if (a == "--logit-band")  logit_band    = (float) atof(next());
        else if (a == "--logit-width") logit_width   = (float) atof(next());
        else if (a == "--upweight")    upweight      = (float) atof(next());
        else if (a == "--n-tokens")    n_tokens      = atoi(next());
        else if (a == "--n-train")     n_train       = atoi(next());
        else if (a == "--n-test")      n_test        = atoi(next());
        else if (a == "--steps")       steps         = atoi(next());
        else if (a == "--accum")       accum         = atoi(next());
        else if (a == "--lr")          lr            = (float) atof(next());
        else if (a == "--skip")        skip          = atoi(next());
        else if (a == "--eval-every")  eval_every    = atoi(next());
        else if (a == "--seed")        seed          = atoi(next());
        else if (a == "--threads")     n_threads     = atoi(next());
        else if (a == "--fd-check")    fd_check      = atoi(next());
        else if (a == "--target-temp") target_temp   = (float) atof(next());
        else if (a == "--layers")      layers_str    = next();
        else if (a == "--lr-frac")     lr_frac       = (float) atof(next());
        else if (a == "--fixed-norm")  fixed_norm    = (float) atof(next());
        else if (a == "--init-vec")    init_vec_path = next();
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    const bool logit_mode = !std::isnan(logit_band);
    if (target_temp > 0 && band_p <= 0 && !logit_mode) band_p = 0.25f; // band only used for reporting
    std::vector<int> layers;
    if (!layers_str.empty()) {
        char * c = strdup(layers_str.c_str());
        for (char * t = strtok(c, ","); t; t = strtok(nullptr, ",")) layers.push_back(atoi(t));
        free(c);
        if (layer < 0) layer = layers[0];
    } else {
        layers.push_back(layer);
    }
    const bool sweep = !layers_str.empty();
    if (model_path.empty() || rollouts_path.empty() || out_path.empty() || layer < 0 || (band_p <= 0 && !logit_mode)) {
        fprintf(stderr, "usage: homeo-steeropt -m model.gguf --rollouts rollouts.jsonl --layer L (--band P | --logit-band Z) --out v.f32 [options]\n");
        return 1;
    }
    const float lband = logit_mode ? logit_band : std::log(band_p);
    const float lwid  = logit_mode ? logit_width : std::log(width);

    // ---- windows from disjoint documents ----
    std::vector<window> train, test;
    {
        struct ro { std::vector<int32_t> prompt; std::string text; int doc; };
        std::vector<ro> rolls;
        std::ifstream f(rollouts_path);
        if (!f) die("cannot open rollouts file");
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const json j = json::parse(line);
            rolls.push_back({ j.at("prompt_ids").get<std::vector<int32_t>>(), j.at("text").get<std::string>(), j.value("doc_idx", 0) });
        }
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        llama_model * vm = llama_model_load_from_file(model_path.c_str(), mp);
        if (!vm) die("vocab load failed");
        const llama_vocab * vocab = llama_model_get_vocab(vm);
        std::mt19937_64 rng(seed);
        std::vector<size_t> order(rolls.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::shuffle(order.begin(), order.end(), rng);
        for (size_t oi : order) {
            const bool is_test = rolls[oi].doc % 4 == 0;
            if ((is_test && (int) test.size() >= n_test) || (!is_test && (int) train.size() >= n_train)) continue;
            std::vector<int32_t> toks = rolls[oi].prompt;
            std::vector<llama_token> tt(rolls[oi].text.size() + 16);
            const int32_t n = llama_tokenize(vocab, rolls[oi].text.c_str(), (int32_t) rolls[oi].text.size(), tt.data(), (int32_t) tt.size(), false, true);
            if (n < 0) die("tokenize failed");
            toks.insert(toks.end(), tt.begin(), tt.begin() + n);
            if ((int) toks.size() < n_tokens) continue;
            const int start = (int) (rng() % (uint64_t) (toks.size() - n_tokens + 1));
            window w;
            w.toks.assign(toks.begin() + start, toks.begin() + start + n_tokens);
            (is_test ? test : train).push_back(std::move(w));
        }
        llama_model_free(vm);
        if (train.empty() || test.empty()) die("not enough windows");
        fprintf(stderr, "homeo-steeropt: %zu train / %zu test windows of %d tokens\n", train.size(), test.size(), n_tokens);
    }

    // ---- model + graph ----
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(0);
    if (!backend) die("cuda init failed");
    jlens_model model;
    if (!jlens_model_load(model_path.c_str(), buft, model)) die("model load failed");
    const int64_t n_embd = model.hparams.n_embd;
    if (layer >= model.hparams.n_layer) die("--layer out of range");

    for (int il : layers) if (il < 0 || il >= model.hparams.n_layer) die("--layers out of range");
    const int V = (int) model.get("output.weight")->ne[1];

    ggml_init_params sparams = { 4 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
    ggml_context * sctx = ggml_init(sparams);
    ggml_tensor * seed_t = ggml_new_tensor_1d(sctx, GGML_TYPE_F32, 1);
    if (!ggml_backend_alloc_ctx_tensors_from_buft(sctx, buft)) die("seed alloc failed");
    const float one = 1.0f;
    ggml_backend_tensor_set(seed_t, &one, 0, sizeof(float));

    // all inputs get storage outside gallocr (see jlens-fit: gallocr aliased leafs there); shared by every layer's graph
    ggml_init_params inparams = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * inctx = ggml_init(inparams);
    ggml_tensor * t_tokens = ggml_new_tensor_1d(inctx, GGML_TYPE_I32, n_tokens);
    ggml_tensor * t_pos    = ggml_new_tensor_1d(inctx, GGML_TYPE_I32, n_tokens);
    ggml_tensor * t_mask   = ggml_new_tensor_2d(inctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_tensor * t_labels = ggml_new_tensor_2d(inctx, GGML_TYPE_F32, V, n_tokens);
    ggml_tensor * t_steer  = ggml_new_tensor_1d(inctx, GGML_TYPE_F32, n_embd);
    if (!ggml_backend_alloc_ctx_tensors_from_buft(inctx, buft)) die("input alloc failed");
    {
        std::vector<int32_t> posv(n_tokens);
        for (int i = 0; i < n_tokens; ++i) posv[i] = i;
        std::vector<float> mask((size_t) n_tokens * n_tokens);
        for (int j = 0; j < n_tokens; ++j)
            for (int i = 0; i < n_tokens; ++i)
                mask[i + (size_t) j * n_tokens] = (i <= j) ? 0.0f : -FLT_MAX;
        ggml_backend_tensor_set(t_pos,  posv.data(), 0, posv.size() * sizeof(int32_t));
        ggml_backend_tensor_set(t_mask, mask.data(), 0, mask.size() * sizeof(float));
    }

    jlens_steer_forward fwd;
    ggml_cgraph *  gb     = nullptr;
    ggml_tensor *  grad_v = nullptr;
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    auto setup = [&](int L) {
        if (fwd.gctx) ggml_free(fwd.gctx);
        fwd = jlens_steer_forward();
        if (!jlens_build_forward_steer(model, n_tokens, L, fwd)) die("graph build failed");
        std::vector<ggml_tensor *> grad_accs(ggml_graph_n_nodes(fwd.gf), nullptr);
        bool found = false;
        for (int i = 0; i < ggml_graph_n_nodes(fwd.gf); ++i) {
            if (ggml_graph_node(fwd.gf, i) == fwd.loss) { grad_accs[i] = seed_t; found = true; break; }
        }
        if (!found) die("loss node not found");
        ggml_set_output(fwd.l_out[L]); // read once per layer for the lr scale
        gb = ggml_graph_dup(fwd.gctx, fwd.gf, /*force_grads=*/true);
        ggml_build_backward_expand(fwd.gctx, gb, grad_accs.data());
        grad_v = ggml_graph_get_grad(gb, fwd.steer);
        if (!grad_v) die("no grad for the steering vector");
        ggml_set_output(grad_v);
        auto wire = [](ggml_tensor * dst, const ggml_tensor * src) { dst->data = src->data; dst->buffer = src->buffer; };
        wire(fwd.tokens, t_tokens); wire(fwd.pos, t_pos); wire(fwd.mask, t_mask);
        wire(fwd.labels, t_labels); wire(fwd.steer, t_steer);
        if (!ggml_gallocr_alloc_graph(galloc, gb)) die("galloc failed");
        fprintf(stderr, "homeo-steeropt: layer %d graph nodes fwd=%d fwd+bwd=%d, compute buffer %.2f GiB\n", L,
                ggml_graph_n_nodes(fwd.gf), ggml_graph_n_nodes(gb), ggml_gallocr_get_buffer_size(galloc, 0) / (1024.0 * 1024.0 * 1024.0));
    };
    setup(layers[0]);

    std::vector<float> v(n_embd, 0.0f), logits((size_t) V * n_tokens);
    auto run = [&](const window & w, bool with_labels) -> float {
        ggml_backend_tensor_set(fwd.tokens, w.toks.data(), 0, n_tokens * sizeof(int32_t));
        ggml_backend_tensor_set(fwd.steer, v.data(), 0, n_embd * sizeof(float));
        if (with_labels) ggml_backend_tensor_set(fwd.labels, w.r.data(), 0, w.r.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, gb) != GGML_STATUS_SUCCESS) die("compute failed");
        float loss = 0.0f;
        ggml_backend_tensor_get(fwd.loss, &loss, 0, sizeof(float));
        return loss;
    };

    // ---- unsteered distributions and targets ----
    auto in_band = [&](float lp, float lse) { return std::fabs((logit_mode ? lp + lse : lp) - lband) <= lwid; };
    auto prepare = [&](window & w) {
        w.r.assign((size_t) V * n_tokens, 1.0f / V); // any valid labels for the v = 0 pass
        run(w, true);
        ggml_backend_tensor_get(fwd.logits, logits.data(), 0, logits.size() * sizeof(float));
        w.lp.resize((size_t) V * n_tokens);
        w.lse.resize(n_tokens);
        w.has_band.assign(n_tokens, 0);
        parallel_for(n_tokens, n_threads, [&](int t) {
            const float * lp = w.lp.data() + (size_t) t * V;
            w.lse[t] = log_softmax(logits.data() + (size_t) t * V, w.lp.data() + (size_t) t * V, V);
            float * r = w.r.data() + (size_t) t * V;
            double z = 0.0;
            bool any = false;
            for (int x = 0; x < V; ++x) {
                const bool b = t >= skip && in_band(lp[x], w.lse[t]);
                any |= b;
                if (target_temp > 0) {
                    r[x] = t >= skip ? std::exp(lp[x] / target_temp) : std::exp(lp[x]);
                } else {
                    r[x] = std::exp(lp[x]) * (b ? upweight : 1.0f);
                }
                z += r[x];
            }
            for (int x = 0; x < V; ++x) r[x] = (float) (r[x] / z);
            w.has_band[t] = any;
        });
    };
    for (auto & w : train) prepare(w);
    for (auto & w : test)  prepare(w);

    if (fd_check > 0) {
        // central differences of the window loss along random unit directions, at v = 0 and at a random v
        std::mt19937_64 frng(seed + 7);
        std::normal_distribution<float> nd;
        std::vector<float> g(n_embd), d(n_embd), v0(n_embd);
        for (int base = 0; base < 2; ++base) {
            for (auto & x : v0) x = base ? 20.0f * nd(frng) / std::sqrt((float) n_embd) * 10.0f : 0.0f;
            v = v0;
            run(train[0], true);
            ggml_backend_tensor_get(grad_v, g.data(), 0, n_embd * sizeof(float));
            for (int k = 0; k < fd_check; ++k) {
                double dn = 0.0;
                for (int i = 0; i < n_embd; ++i) { d[i] = k == 0 ? g[i] : nd(frng); dn += (double) d[i] * d[i]; } // dir 0: gradient direction
                for (auto & x : d) x /= (float) std::sqrt(dn);
                double gd = 0.0;
                for (int i = 0; i < n_embd; ++i) gd += (double) g[i] * d[i];
                for (float h : { 32.0f, 16.0f, 8.0f }) {
                    for (int i = 0; i < n_embd; ++i) v[i] = v0[i] + h * d[i];
                    const double lp_ = run(train[0], true);
                    for (int i = 0; i < n_embd; ++i) v[i] = v0[i] - h * d[i];
                    const double lm_ = run(train[0], true);
                    fprintf(stderr, "fd %s dir %d h=%.1f: grad.d %+.6e  fd %+.6e  ratio %.4f\n",
                            base ? "v=rand" : "v=0   ", k, h, gd, (lp_ - lm_) / (2 * h), (lp_ - lm_) / (2 * h) / gd);
                }
            }
        }
        return 0;
    }

    // test metrics for a distribution given by per-row log q
    auto metrics = [&](const window & w, const std::vector<float> & lq_all, row_metrics & acc) {
        std::vector<row_metrics> rm(n_tokens);
        parallel_for(n_tokens, n_threads, [&](int t) {
            if (t < skip) return;
            const float * lp = w.lp.data() + (size_t) t * V;
            const float * lq = lq_all.data() + (size_t) t * V;
            const float * r  = w.r.data() + (size_t) t * V;
            row_metrics & m = rm[t];
            int top_p = 0, top_q = 0;
            double bp = 0, bq = 0, np_ = 0, nq = 0, kl = 0;
            for (int x = 0; x < V; ++x) {
                m.ce -= r[x] * lq[x];
                if (lp[x] > lp[top_p]) top_p = x;
                if (lq[x] > lq[top_q]) top_q = x;
                const double p = std::exp(lp[x]), q = std::exp(lq[x]);
                if (in_band(lp[x], w.lse[t])) { bp += p; bq += q; } else { np_ += p; nq += q; }
            }
            if (w.has_band[t]) {
                for (int x = 0; x < V; ++x) {
                    if (in_band(lp[x], w.lse[t])) continue;
                    const double p = std::exp(lp[x]) / np_;
                    if (p > 0) kl += p * (lp[x] - std::log(np_) - (lq[x] - std::log(nq)));
                }
                m.band_ratio = bq / bp; m.kl_nonband = kl; m.n_band = 1;
            }
            m.top_changed = top_p != top_q;
            m.n = 1;
        });
        for (auto & m : rm) {
            acc.ce += m.ce; acc.band_ratio += m.band_ratio; acc.kl_nonband += m.kl_nonband;
            acc.top_changed += m.top_changed; acc.n += m.n; acc.n_band += m.n_band;
        }
    };
    auto summarize = [](const row_metrics & a) {
        json j;
        j["ce"] = a.ce / a.n; j["band_ratio"] = a.n_band ? a.band_ratio / a.n_band : 0.0;
        j["kl_nonband"] = a.n_band ? a.kl_nonband / a.n_band : 0.0; j["top_changed"] = a.top_changed / a.n;
        j["band_rows_frac"] = (double) a.n_band / a.n;
        return j;
    };

    // reference rows: v = 0, the target itself (CE floor), and the best global temperature
    json res;
    double h_r = 0.0; int n_rows = 0;
    row_metrics base;
    for (auto & w : test) {
        metrics(w, w.lp, base);
        for (int t = skip; t < n_tokens; ++t) {
            const float * r = w.r.data() + (size_t) t * V;
            for (int x = 0; x < V; ++x) if (r[x] > 0) h_r -= r[x] * std::log(r[x]);
            n_rows++;
        }
    }
    h_r /= n_rows;
    res["test_v0"] = summarize(base);
    res["ce_floor_H_r"] = h_r;
    {
        double best_ce = 1e30, best_T = 1.0;
        json bestj;
        for (double T = 0.5; T <= 2.51; T += 0.05) {
            row_metrics m;
            for (auto & w : test) {
                std::vector<float> lq((size_t) V * n_tokens);
                parallel_for(n_tokens, n_threads, [&](int t) {
                    std::vector<float> zt(V);
                    for (int x = 0; x < V; ++x) zt[x] = (float) (w.lp[(size_t) t * V + x] / T);
                    log_softmax(zt.data(), lq.data() + (size_t) t * V, V);
                });
                metrics(w, lq, m);
            }
            if (m.ce / m.n < best_ce) { best_ce = m.ce / m.n; best_T = T; bestj = summarize(m); }
        }
        bestj["T"] = best_T;
        res["test_best_temperature"] = bestj;
    }
    fprintf(stderr, "test: CE v=0 %.5f | best temperature T=%.2f CE %.5f | floor H(r) %.5f\n",
            res["test_v0"]["ce"].get<double>(), res["test_best_temperature"]["T"].get<double>(),
            res["test_best_temperature"]["ce"].get<double>(), h_r);

    auto eval_v = [&]() {
        row_metrics m;
        std::vector<float> lq((size_t) V * n_tokens);
        for (auto & w : test) {
            run(w, true);
            ggml_backend_tensor_get(fwd.logits, logits.data(), 0, logits.size() * sizeof(float));
            parallel_for(n_tokens, n_threads, [&](int t) { log_softmax(logits.data() + (size_t) t * V, lq.data() + (size_t) t * V, V); });
            metrics(w, lq, m);
        }
        return summarize(m);
    };

    res["band"] = logit_mode ? logit_band : band_p; res["band_kind"] = logit_mode ? "raw_logit" : "p";
    res["target_temp"] = target_temp;

    // ---- Adam, one vector per layer ----
    for (size_t li = 0; li < layers.size(); ++li) {
    const int L = layers[li];
    if (li > 0) setup(L);
    std::fill(v.begin(), v.end(), 0.0f);
    double lr_L = lr;
    double h_norm = 0.0;
    {
        int nr = 0;
        std::vector<float> h((size_t) n_embd * n_tokens);
        for (int k = 0; k < std::min<int>(4, (int) train.size()); ++k) {
            run(train[k], true);
            ggml_backend_tensor_get(fwd.l_out[L], h.data(), 0, h.size() * sizeof(float));
            for (int t = skip; t < n_tokens; ++t) {
                double s2 = 0.0;
                for (int e = 0; e < n_embd; ++e) s2 += (double) h[(size_t) t * n_embd + e] * h[(size_t) t * n_embd + e];
                h_norm += std::sqrt(s2); nr++;
            }
        }
        h_norm /= nr;
        if (lr_frac > 0) lr_L = lr_frac * h_norm / std::sqrt((double) n_embd);
    }
    const double radius = fixed_norm * h_norm;
    auto renorm = [&]() { // put v back on the sphere |v| = radius
        double n2 = 0.0;
        for (float x : v) n2 += (double) x * x;
        const double f = radius / std::sqrt(std::max(n2, 1e-30));
        for (auto & x : v) x = (float) (x * f);
    };
    if (fixed_norm > 0) {
        if (!init_vec_path.empty()) {
            FILE * f = fopen(init_vec_path.c_str(), "rb");
            if (!f || fread(v.data(), sizeof(float), n_embd, f) != (size_t) n_embd) die("cannot read --init-vec");
            fclose(f);
        } else {
            std::mt19937_64 irng(seed + 3);
            std::normal_distribution<float> nd;
            for (auto & x : v) x = nd(irng);
        }
        renorm();
    }
    fprintf(stderr, "== layer %d: mean |h| %.1f, lr %.4f%s\n", L, h_norm, lr_L, fixed_norm > 0 ? " (fixed norm)" : "");
    json lres;
    lres["layer"] = L; lres["resid_norm"] = h_norm; lres["lr"] = lr_L;
    if (fixed_norm > 0) { lres["fixed_norm"] = fixed_norm; lres["init_vec"] = init_vec_path; }
    if (fixed_norm > 0) { // test CE at the starting point, so the search can be compared with where it began
        json e0 = eval_v(); e0["step"] = 0; e0["v_norm"] = radius;
        lres["test_eval"].push_back(e0);
        fprintf(stderr, "  test (start): CE %.5f  band ratio %.3f\n", e0["ce"].get<double>(), e0["band_ratio"].get<double>());
    }
    std::vector<double> m1(n_embd, 0.0), m2(n_embd, 0.0);
    std::vector<float> g(n_embd), gacc(n_embd);
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    std::mt19937_64 rng(seed + 1);
    std::vector<size_t> tr_order(train.size());
    for (size_t i = 0; i < tr_order.size(); ++i) tr_order[i] = i;
    size_t tr_pos = tr_order.size();
    const auto t0 = std::chrono::steady_clock::now();
    for (int step = 1; step <= steps; ++step) {
        std::fill(gacc.begin(), gacc.end(), 0.0f);
        double loss = 0.0;
        for (int a = 0; a < accum; ++a) {
            if (tr_pos >= tr_order.size()) { std::shuffle(tr_order.begin(), tr_order.end(), rng); tr_pos = 0; }
            loss += run(train[tr_order[tr_pos++]], true) / accum;
            ggml_backend_tensor_get(grad_v, g.data(), 0, n_embd * sizeof(float));
            for (int i = 0; i < n_embd; ++i) gacc[i] += g[i] / accum;
        }
        if (fixed_norm > 0) { // tangent-space gradient: the radial part only changes |v|, which is fixed
            double gv = 0.0, vv = 0.0;
            for (int i = 0; i < n_embd; ++i) { gv += (double) gacc[i] * v[i]; vv += (double) v[i] * v[i]; }
            for (int i = 0; i < n_embd; ++i) gacc[i] -= (float) (gv / vv * v[i]);
        }
        double vn = 0.0;
        for (int i = 0; i < n_embd; ++i) {
            m1[i] = b1 * m1[i] + (1 - b1) * gacc[i];
            m2[i] = b2 * m2[i] + (1 - b2) * (double) gacc[i] * gacc[i];
            const double mh = m1[i] / (1 - std::pow(b1, step)), vh = m2[i] / (1 - std::pow(b2, step));
            v[i] -= (float) (lr_L * mh / (std::sqrt(vh) + eps));
            vn += (double) v[i] * v[i];
        }
        if (fixed_norm > 0) { renorm(); vn = radius * radius; }
        if (step % 10 == 0 || step == 1) {
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "step %4d  train CE %.5f  |v| %.1f  (%.1f s)\n", step, loss, std::sqrt(vn), el);
        }
        if (step % eval_every == 0 || step == steps) {
            double tr_ce = 0.0;
            const int n_fix = std::min<int>(8, (int) train.size());
            for (int k = 0; k < n_fix; ++k) tr_ce += run(train[k], true) / n_fix;
            json e = eval_v();
            e["train_fixed_ce"] = tr_ce;
            fprintf(stderr, "  train (fixed %d windows): CE %.5f\n", n_fix, tr_ce);
            e["step"] = step; e["v_norm"] = std::sqrt(vn);
            lres["test_eval"].push_back(e);
            fprintf(stderr, "  test: CE %.5f  band ratio %.3f  KL(nonband) %.4f  top changed %.3f\n",
                    e["ce"].get<double>(), e["band_ratio"].get<double>(), e["kl_nonband"].get<double>(), e["top_changed"].get<double>());
        }
    }


    {
        char name[64];
        snprintf(name, sizeof(name), "_L%02d.f32", L);
        const std::string vpath = sweep ? out_path + name : out_path;
        FILE * f = fopen(vpath.c_str(), "wb");
        if (!f || fwrite(v.data(), sizeof(float), n_embd, f) != (size_t) n_embd) die("cannot write vector");
        fclose(f);
    }
    res["layers"].push_back(lres);
    std::ofstream(out_path + ".json") << res.dump(2) << "\n"; // rewritten after every layer
    }

    res["band"] = logit_mode ? logit_band : band_p; res["band_kind"] = logit_mode ? "raw_logit" : "p";
    res["width"] = logit_mode ? logit_width : width; res["upweight"] = upweight; res["target_temp"] = target_temp; res["steps"] = steps;
    res["lr"] = lr; res["lr_frac"] = lr_frac; res["fixed_norm"] = fixed_norm;
    res["n_train"] = train.size(); res["n_test"] = test.size(); res["n_tokens"] = n_tokens;
    std::ofstream(out_path + ".json") << res.dump(2) << "\n";
    fprintf(stderr, "homeo-steeropt: wrote %s.json and %zu vector(s)\n", out_path.c_str(), layers.size());
    return 0;
}
