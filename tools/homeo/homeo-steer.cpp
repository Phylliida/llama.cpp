// homeo-steer: exact next-token effect of a steering vector (S2 in NOTES-homeostasis.md).
//
// Teacher-forces rollouts with a control vector at one layer (all positions) for
// each scale, and compares the steered next-token distribution q against the
// unsteered p at every --stride-th position. No sampling: all stats are exact.
// Per scale it reports:
//   mass[b]    q mass on tokens whose log10 p falls in bin b (homeo-capture bins)
//   tmass[b]   same for the temperature-scaled p_T (p_T ~ p^(1/T)) matched per
//              position on E_q[-ln p]; mass != tmass means the vector is not just
//              a temperature knob (a band-pass "pick p" vector would move mass
//              off the top bin into a middle band)
//   E_q[-ln p], H(q), KL(q||p), KL(q||p_T), q(argmax p), argmax agreement, mean T
//   per --levels P: q / p_T / p mass on tokens with p in [P/w, P*w] (--width w), or with raw
//   logit z in [Z-d, Z+d] for --logit-levels Z (--logit-width d), and how
//   often the steered argmax is a band token that was not the unsteered argmax (temperature
//   never does this: it keeps the order)
//
// usage: homeo-steer -m model.gguf --rollouts rollouts.jsonl --vec vec.f32 --layer L
//          [--scales -0.4,-0.2,0.2,0.4] [--n-rollouts 16] [--stride 8] [--skip 32]
//          [--max-len 2100] [--levels 0.5,0.25,0.1,0.03] [--width 1.5] [--out result.json]
//          [--logit-levels Z,...] [--logit-width 1] [--dump-raw logits.f16]
// --dump-raw writes the unsteered raw logits at the eval positions (f16 [n_pos, n_vocab]);
// without --vec the tool stops after that pass.
// vec.f32 is raw f32 [n_embd]; the added vector is scale * vec.
// --layer-end E adds the same scale * vec after every block in [--layer, E] (multi-layer steering,
// like control vectors applied over a layer band); default E = --layer.
// --dump-topn N writes, per scale, PREFIX_sSCALE.bin from --dump-prefix: for every eval position the N most
// likely unsteered tokens as (int32 id, f32 log p, f32 log q), for within-token analyses of the steering effect.

#include "llama.h"
#include "ggml.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

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
    fprintf(stderr, "homeo-steer: %s\n", msg);
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

static void log_softmax(const float * z, float * out, int n) {
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) m = std::max(m, z[i]);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp(z[i] - m);
    const float lse = m + (float) std::log(s);
    for (int i = 0; i < n; ++i) out[i] = z[i] - lse;
}

// E_{p_T}[-ln p] and the normalizer of p_T ~ p^(1/T), from log p
static double tempered_surprisal(const float * lp, int n, double T, double * lse_out) {
    double m = -INFINITY;
    for (int i = 0; i < n; ++i) m = std::max(m, (double) lp[i] / T);
    double s = 0.0, sl = 0.0;
    for (int i = 0; i < n; ++i) {
        const double e = std::exp(lp[i] / T - m);
        s  += e;
        sl += e * lp[i];
    }
    if (lse_out) *lse_out = m + std::log(s);
    return -sl / s;
}

static const int MAX_LEVELS = 8;
static std::vector<float> g_llev; // band centers: log p (--levels) or raw logit (--logit-levels)
static float g_lw = std::log(1.5f);
static bool  g_logit_bands = false;

struct pos_stats {
    double mass[N_BINS]  = {};
    double tmass[N_BINS] = {};
    double band_q[MAX_LEVELS] = {}, band_t[MAX_LEVELS] = {}, band_p[MAX_LEVELS] = {}, band_inv[MAX_LEVELS] = {};
    double e_surp = 0, h_q = 0, kl_qp = 0, kl_qt = 0, q_top = 0, agree = 0, T = 0;
};

static pos_stats compare(const float * lp, const float * lq, int n, float lse_p) {
    const float zoff = g_logit_bands ? lse_p : 0.0f; // band key: raw logit = log p + lse
    pos_stats st;
    int top_p = 0, top_q = 0;
    for (int i = 0; i < n; ++i) {
        if (lp[i] > lp[top_p]) top_p = i;
        if (lq[i] > lq[top_q]) top_q = i;
    }
    const float LN10 = std::log(10.0f);
    for (int i = 0; i < n; ++i) {
        const double q = std::exp((double) lq[i]);
        st.mass[lp10_bin(lp[i] / LN10)] += q;
        st.e_surp -= q * lp[i];
        st.h_q    -= q * lq[i];
        st.kl_qp  += q * (lq[i] - lp[i]);
    }
    st.q_top = std::exp((double) lq[top_p]);
    st.agree = top_p == top_q;
    for (size_t b = 0; b < g_llev.size(); ++b) {
        st.band_inv[b] = top_q != top_p && std::fabs(lp[top_q] + zoff - g_llev[b]) <= g_lw;
    }

    // match T on E[-ln p] by bisection in log T; clamp at the ends
    double lo = std::log(0.02), hi = std::log(50.0);
    for (int it = 0; it < 40; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (tempered_surprisal(lp, n, std::exp(mid), nullptr) < st.e_surp) lo = mid; else hi = mid;
    }
    const double T = std::exp(0.5 * (lo + hi));
    double lse;
    tempered_surprisal(lp, n, T, &lse);
    for (int i = 0; i < n; ++i) {
        const double lt = lp[i] / T - lse;
        st.tmass[lp10_bin(lp[i] / LN10)] += std::exp(lt);
        st.kl_qt += std::exp((double) lq[i]) * (lq[i] - lt);
        for (size_t b = 0; b < g_llev.size(); ++b) {
            if (std::fabs(lp[i] + zoff - g_llev[b]) <= g_lw) {
                st.band_q[b] += std::exp((double) lq[i]);
                st.band_t[b] += std::exp(lt);
                st.band_p[b] += std::exp((double) lp[i]);
            }
        }
    }
    st.T = T;
    return st;
}

int main(int argc, char ** argv) {
    std::string model_path, rollouts_path, vec_path, out_path, scales_str = "-0.4,-0.2,0.2,0.4", levels_str = "0.5,0.25,0.1,0.03";
    float width = 1.5f, logit_width = 1.0f;
    std::string logit_levels_str, dump_raw_path, dump_prefix;
    int dump_topn = 0;
    int layer = -1, layer_end = -1, n_rollouts = 16, stride = 8, skip = 32, max_len = 2100, ubatch = 512;
    int n_threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) die(("missing value for " + a).c_str()); return argv[++i]; };
        if      (a == "-m")           model_path    = next();
        else if (a == "--rollouts")   rollouts_path = next();
        else if (a == "--vec")        vec_path      = next();
        else if (a == "--layer")      layer         = atoi(next());
        else if (a == "--layer-end")  layer_end     = atoi(next());
        else if (a == "--scales")     scales_str    = next();
        else if (a == "--n-rollouts") n_rollouts    = atoi(next());
        else if (a == "--stride")     stride        = atoi(next());
        else if (a == "--skip")       skip          = atoi(next());
        else if (a == "--max-len")    max_len       = atoi(next());
        else if (a == "--threads")    n_threads     = atoi(next());
        else if (a == "--out")        out_path      = next();
        else if (a == "--levels")     levels_str    = next();
        else if (a == "--width")      width         = (float) atof(next());
        else if (a == "--logit-levels") logit_levels_str = next();
        else if (a == "--logit-width")  logit_width   = (float) atof(next());
        else if (a == "--dump-raw")     dump_raw_path = next();
        else if (a == "--dump-topn")    dump_topn     = atoi(next());
        else if (a == "--dump-prefix")  dump_prefix   = next();
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || rollouts_path.empty() || ((vec_path.empty() || layer < 1) && dump_raw_path.empty())) {
        fprintf(stderr, "usage: homeo-steer -m model.gguf --rollouts rollouts.jsonl --vec vec.f32 --layer L (>= 1) [options]\n");
        return 1;
    }
    std::vector<float> scales;
    {
        char * s = strdup(scales_str.c_str());
        for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) scales.push_back((float) atof(t));
        free(s);
    }

    std::vector<float> levels;
    g_logit_bands = !logit_levels_str.empty();
    {
        char * s = strdup((g_logit_bands ? logit_levels_str : levels_str).c_str());
        for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) levels.push_back((float) atof(t));
        free(s);
    }
    if (levels.size() > (size_t) MAX_LEVELS) die("at most 8 --levels");
    for (float x : levels) g_llev.push_back(g_logit_bands ? x : std::log(x));
    g_lw = g_logit_bands ? logit_width : std::log(width);

    std::vector<std::pair<std::vector<int32_t>, std::string>> rollouts;
    {
        std::ifstream f(rollouts_path);
        if (!f) die("cannot open rollouts file");
        std::string line;
        while (std::getline(f, line) && (int) rollouts.size() < n_rollouts) {
            if (line.empty()) continue;
            const json j = json::parse(line);
            rollouts.push_back({ j.at("prompt_ids").get<std::vector<int32_t>>(), j.at("text").get<std::string>() });
        }
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) die("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    if (layer >= n_layer) die("--layer out of range");
    if (layer_end < layer) layer_end = layer;
    if (layer_end >= n_layer) die("--layer-end out of range");
    const bool dump_only = vec_path.empty();

    std::vector<float> vec(n_embd);
    if (!dump_only) {
        FILE * f = fopen(vec_path.c_str(), "rb");
        if (!f || fread(vec.data(), sizeof(float), n_embd, f) != (size_t) n_embd) die("cannot read --vec (raw f32 [n_embd])");
        fclose(f);
    }
    double vnorm = 0.0;
    for (float x : vec) vnorm += (double) x * x;
    vnorm = std::sqrt(vnorm);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = max_len + 256;
    cp.n_batch         = ubatch;
    cp.n_ubatch        = ubatch;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.no_perf         = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context init failed");
    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<std::vector<int32_t>> toks(rollouts.size());
    for (size_t r = 0; r < rollouts.size(); ++r) {
        toks[r] = rollouts[r].first;
        const std::string & text = rollouts[r].second;
        std::vector<llama_token> tt(text.size() + 16);
        const int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tt.data(), (int32_t) tt.size(), false, true);
        if (n < 0) die("tokenize failed");
        toks[r].insert(toks[r].end(), tt.begin(), tt.begin() + n);
        if ((int) toks[r].size() > max_len) toks[r].resize(max_len);
    }

    // log-probs at eval positions for one full pass (current cvec)
    llama_batch batch = llama_batch_init(ubatch, 0, 1);
    std::vector<float> base_lse;
    FILE * f_raw = nullptr;
    auto run_pass = [&](std::vector<std::vector<float>> & out, std::vector<float> * lse_out) {
        out.clear();
        if (lse_out) lse_out->clear();
        for (auto & tk : toks) {
            llama_memory_clear(mem, true);
            const int len = (int) tk.size();
            for (int cur = 0; cur < len; cur += ubatch) {
                const int n = std::min(ubatch, len - cur);
                batch.n_tokens = n;
                std::vector<int> rows;
                for (int j = 0; j < n; ++j) {
                    const int i = cur + j;
                    batch.token[j] = tk[i]; batch.pos[j] = i; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
                    batch.logits[j] = i >= skip && (i - skip) % stride == 0 && i + 1 < len;
                    if (batch.logits[j]) rows.push_back(j);
                }
                if (llama_decode(ctx, batch) != 0) die("llama_decode failed");
                const size_t base = out.size();
                out.resize(base + rows.size(), std::vector<float>(n_vocab));
                std::vector<const float *> zr(rows.size());
                for (size_t k = 0; k < rows.size(); ++k) zr[k] = llama_get_logits_ith(ctx, rows[k]);
                parallel_for((int) rows.size(), n_threads, [&](int k) {
                    log_softmax(zr[k], out[base + k].data(), n_vocab);
                });
                for (size_t k = 0; k < rows.size(); ++k) {
                    if (lse_out) lse_out->push_back(zr[k][0] - out[base + k][0]);
                    if (f_raw) {
                        std::vector<ggml_fp16_t> h(n_vocab);
                        ggml_fp32_to_fp16_row(zr[k], h.data(), n_vocab);
                        fwrite(h.data(), sizeof(ggml_fp16_t), n_vocab, f_raw);
                    }
                }
            }
        }
    };

    std::vector<float> cvec((size_t) n_embd * n_layer, 0.0f);
    std::vector<std::vector<float>> base_lp, steer_lp;
    if (llama_set_adapter_cvec(ctx, nullptr, 0, n_embd, 0, 0) != 0) die("cvec clear failed");
    if (!dump_raw_path.empty()) {
        f_raw = fopen(dump_raw_path.c_str(), "wb");
        if (!f_raw) die("cannot open --dump-raw file");
    }
    run_pass(base_lp, &base_lse);
    if (f_raw) {
        fclose(f_raw);
        f_raw = nullptr;
        json jm;
        jm["n_positions"] = base_lp.size(); jm["n_vocab"] = n_vocab; jm["n_rollouts"] = rollouts.size();
        jm["stride"] = stride; jm["skip"] = skip; jm["dtype"] = "f16";
        std::ofstream(dump_raw_path + ".json") << jm.dump(2) << "\n";
        fprintf(stderr, "homeo-steer: wrote raw logits [%zu, %d] -> %s\n", base_lp.size(), n_vocab, dump_raw_path.c_str());
        if (dump_only) return 0;
    }
    fprintf(stderr, "homeo-steer: baseline pass, %zu eval positions, |vec| = %.3f\n", base_lp.size(), vnorm);

    json res;
    res["vec"] = vec_path; res["layer"] = layer; res["layer_end"] = layer_end; res["vec_norm"] = vnorm;
    res["levels"] = levels; res["width"] = g_logit_bands ? logit_width : width;
    res["band_kind"] = g_logit_bands ? "raw_logit" : "log_p";
    res["n_positions"] = base_lp.size(); res["n_rollouts"] = rollouts.size();
    res["bin_edges_log10p"] = std::vector<float>(std::begin(BIN_EDGES), std::end(BIN_EDGES));
    {
        // unsteered mass profile for reference
        std::vector<double> mass(N_BINS, 0.0);
        const float LN10 = std::log(10.0f);
        for (auto & lp : base_lp) for (int i = 0; i < n_vocab; ++i) mass[lp10_bin(lp[i] / LN10)] += std::exp((double) lp[i]);
        for (auto & m : mass) m /= base_lp.size();
        res["base_mass"] = mass;
    }

    for (float sc : scales) {
        for (int il = layer; il <= layer_end; ++il) {
            for (int e = 0; e < n_embd; ++e) cvec[(size_t) (il - 1) * n_embd + e] = sc * vec[e];
        }
        if (llama_set_adapter_cvec(ctx, cvec.data(), cvec.size(), n_embd, layer, layer_end) != 0) die("cvec set failed");
        run_pass(steer_lp, nullptr);

        const int P = (int) base_lp.size();
        std::vector<pos_stats> ps(P);
        parallel_for(P, n_threads, [&](int k) { ps[k] = compare(base_lp[k].data(), steer_lp[k].data(), n_vocab, base_lse[k]); });

        pos_stats avg;
        for (auto & s : ps) {
            for (int b = 0; b < N_BINS; ++b) { avg.mass[b] += s.mass[b] / P; avg.tmass[b] += s.tmass[b] / P; }
            avg.e_surp += s.e_surp / P; avg.h_q += s.h_q / P; avg.kl_qp += s.kl_qp / P; avg.kl_qt += s.kl_qt / P;
            avg.q_top += s.q_top / P; avg.agree += s.agree / P; avg.T += std::log(s.T) / P;
            for (size_t b = 0; b < levels.size(); ++b) {
                avg.band_q[b] += s.band_q[b] / P; avg.band_t[b] += s.band_t[b] / P;
                avg.band_p[b] += s.band_p[b] / P; avg.band_inv[b] += s.band_inv[b] / P;
            }
        }
        json js;
        js["scale"] = sc; js["added_norm"] = std::fabs(sc) * vnorm;
        js["mass"]  = std::vector<double>(avg.mass, avg.mass + N_BINS);
        js["tmass"] = std::vector<double>(avg.tmass, avg.tmass + N_BINS);
        js["E_surprisal"] = avg.e_surp; js["H_q"] = avg.h_q; js["KL_q_p"] = avg.kl_qp; js["KL_q_pT"] = avg.kl_qt;
        js["q_of_p_argmax"] = avg.q_top; js["argmax_agree"] = avg.agree; js["T_geomean"] = std::exp(avg.T);
        for (size_t b = 0; b < levels.size(); ++b) {
            json jb;
            jb["level"] = levels[b]; jb["mass_q"] = avg.band_q[b]; jb["mass_pT"] = avg.band_t[b];
            jb["mass_p"] = avg.band_p[b]; jb["argmax_inverted_into_band"] = avg.band_inv[b];
            js["bands"].push_back(jb);
        }
        res["scales"].push_back(js);

        if (dump_topn > 0 && !dump_prefix.empty()) {
            char name[64];
            snprintf(name, sizeof(name), "_s%g.bin", sc);
            FILE * fd = fopen((dump_prefix + name).c_str(), "wb");
            if (!fd) die("cannot open --dump-prefix file");
            std::vector<int> idx(n_vocab);
            for (int k = 0; k < P; ++k) {
                const float * lp = base_lp[k].data();
                const float * lq = steer_lp[k].data();
                for (int i = 0; i < n_vocab; ++i) idx[i] = i;
                std::partial_sort(idx.begin(), idx.begin() + dump_topn, idx.end(), [&](int a, int b) { return lp[a] > lp[b]; });
                for (int j = 0; j < dump_topn; ++j) {
                    const int32_t id = idx[j];
                    const float rec[2] = { lp[id], lq[id] };
                    fwrite(&id, sizeof(int32_t), 1, fd);
                    fwrite(rec, sizeof(float), 2, fd);
                }
            }
            fclose(fd);
            res["dump_topn"] = dump_topn; res["dump_prefix"] = dump_prefix;
        }

        fprintf(stderr, "scale %+.3f (|add| %.1f): E[-ln p] %.3f  H(q) %.3f  KL(q||p) %.3f  KL(q||p_T) %.3f  q(top) %.3f  agree %.3f  T %.2f\n",
                sc, std::fabs(sc) * vnorm, avg.e_surp, avg.h_q, avg.kl_qp, avg.kl_qt, avg.q_top, avg.agree, std::exp(avg.T));
        fprintf(stderr, "   mass  :"); for (int b = 0; b < N_BINS; ++b) fprintf(stderr, " %.3f", avg.mass[b]);  fprintf(stderr, "\n");
        fprintf(stderr, "   T-mass:"); for (int b = 0; b < N_BINS; ++b) fprintf(stderr, " %.3f", avg.tmass[b]); fprintf(stderr, "\n");
        for (size_t b = 0; b < levels.size(); ++b) {
            fprintf(stderr, "   band p~%.3g: mass q %.4f  p_T %.4f  p %.4f  argmax->band %.4f\n",
                    levels[b], avg.band_q[b], avg.band_t[b], avg.band_p[b], avg.band_inv[b]);
        }
    }

    if (!out_path.empty()) std::ofstream(out_path) << res.dump(2) << "\n";

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
