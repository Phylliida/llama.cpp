// homeo-value-exact: check and time jlens_value_exact (the exact value lens over a whole text).
//
// Check: on a short window, compare against the brute force of homeo-value (one full backward pass of l_t per
// target t, which gives d l_t / d h_s for every s <= t at the slot layer):
//   now[s] = d l_s / d h_s,  fut[s] = sum over t > s of d l_t / d h_s
// Bench: time jlens_value_exact on a long window and print the norms by position.
//
// usage: homeo-value-exact -m model.gguf --ids ids.bin [--offset 0] [--n-tokens 96] [--chunk 16] [--layer 40]
//          [--bench N] [--bench-chunk 64] [--fwd-check N] [--branch-check N [--branch-pos 12] [--fd-eps 0.1]]
//
// Branch check: jlens_branch_grad on N tokens; at --branch-pos positions, for each top-8 candidate c, compare the
// linearized delta with a central finite difference along E[c] - E[actual] (checks the gradient) and with the
// actual swap (checks the linearization).

#include "jlens.h"
#include "llama.h"

#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

static void die(const char * msg) {
    fprintf(stderr, "homeo-value-exact: %s\n", msg);
    exit(1);
}

int main(int argc, char ** argv) {
    std::string model_path, ids_path;
    int T = 96, chunk = 32, L = 40, bench = 0, bench_chunk = 64, fwd_check = 0, branch_check = 0, branch_pos = 12, group_check = 0;
    float fd_eps = 0.1f;
    int pause_ms = 0;
    size_t offset = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) die(("missing value for " + a).c_str()); return argv[++i]; };
        if      (a == "-m")            model_path  = next();
        else if (a == "--ids")         ids_path    = next();
        else if (a == "--offset")      offset      = (size_t) atoll(next());
        else if (a == "--n-tokens")    T           = atoi(next());
        else if (a == "--chunk")       chunk       = atoi(next());
        else if (a == "--layer")       L           = atoi(next());
        else if (a == "--bench")       bench       = atoi(next());
        else if (a == "--bench-chunk") bench_chunk = atoi(next());
        else if (a == "--fwd-check")   fwd_check   = atoi(next());
        else if (a == "--branch-check") branch_check = atoi(next());
        else if (a == "--branch-pos")  branch_pos  = atoi(next());
        else if (a == "--fd-eps")      fd_eps      = (float) atof(next());
        else if (a == "--pause-ms")    pause_ms    = atoi(next());
        else if (a == "--group-check") group_check = atoi(next());
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || ids_path.empty()) die("usage: homeo-value-exact -m model.gguf --ids ids.bin [options]");

    std::vector<int32_t> ids;
    {
        std::ifstream f(ids_path, std::ios::binary | std::ios::ate);
        if (!f) die("cannot open --ids");
        ids.resize((size_t) f.tellg() / sizeof(int32_t));
        f.seekg(0);
        f.read((char *) ids.data(), ids.size() * sizeof(int32_t));
    }
    if (ids.size() < offset + std::max(T, bench)) die("token stream too short");

    llama_backend_init();
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(0);
    if (!backend) die("cuda init failed");
    jlens_model model;
    if (!jlens_model_load(model_path.c_str(), buft, model)) die("model load failed");
    const int64_t n_embd = model.hparams.n_embd;
    const int V = (int) model.get("output.weight")->ne[1];

    std::vector<int32_t> tk(ids.begin() + offset, ids.begin() + offset + T);

    // ---- group check: embedding gradient chained over layer groups of 21 blocks vs one group of all blocks
    if (group_check > 0) {
        std::vector<int32_t> gt(ids.begin() + offset, ids.begin() + offset + group_check);
        jlens_branch_result r1, r2;
        std::vector<float> g1, g2;
        std::string e2;
        if (!jlens_branch_grad(model, backend, buft, gt, 4, chunk, (int) model.hparams.n_vocab, r1, e2, &g1, 21)) die(e2.c_str());
        if (!jlens_branch_grad(model, backend, buft, gt, 4, chunk, (int) model.hparams.n_vocab, r2, e2, &g2, 61)) die(e2.c_str());
        std::vector<double> rel;
        for (int s = 0; s + 1 < group_check; ++s) {
            double d2 = 0.0, a2 = 0.0;
            for (int64_t e = 0; e < n_embd; ++e) {
                const double x = g2[(size_t) s * n_embd + e], y = g1[(size_t) s * n_embd + e];
                d2 += (x - y) * (x - y); a2 += x * x;
            }
            rel.push_back(std::sqrt(d2 / std::max(a2, 1e-30)));
        }
        std::sort(rel.begin(), rel.end());
        fprintf(stderr, "chained (21-block groups) vs one group: embedding gradient rel err median %.2e, max %.2e (%d tokens, chunk %d)\n",
                rel[rel.size() / 2], rel.back(), group_check, r1.chunk);
        return 0;
    }

    // ---- branch check
    if (branch_check > 0) {
        const int NT = branch_check, K = 8;
        const int n_real = (int) model.hparams.n_vocab;
        std::vector<int32_t> bt(ids.begin() + offset, ids.begin() + offset + NT);
        jlens_branch_result br;
        std::vector<float> gemb;
        std::string e2;
        if (!jlens_branch_grad(model, backend, buft, bt, K, chunk, n_real, br, e2, &gemb)) die(e2.c_str());
        fprintf(stderr, "branch: %d tokens, chunk %d: forward %.2f s, backward (3 layer groups) %.2f s; output rows %lld, real vocab %d\n",
                NT, br.chunk, br.sec_fwd, br.sec_bwd, (long long) model.get("output.weight")->ne[1], n_real);
        std::vector<double> ll0;
        if (!jlens_text_loglik(model, backend, buft, bt, chunk, n_real, nullptr, ll0, e2)) die(e2.c_str());
        auto later = [&](const std::vector<double> & ll, int s) { double f = 0.0; for (int t = s + 1; t + 1 < NT; ++t) f += ll[t]; return f; };
        std::vector<double> v_lin, v_fd, v_swap, v_dlin, sp_lin, sp_swap;
        std::vector<float> add((size_t) NT * n_embd, 0.0f);
        for (int pi = 0; pi < branch_pos; ++pi) {
            const int s = 4 + (int) ((int64_t) pi * (NT - 48) / std::max(1, branch_pos - 1));
            const int32_t act = bt[s + 1];
            std::vector<int32_t> rid = { act };
            for (int r = 0; r < K; ++r) rid.push_back(br.cand[(size_t) s * K + r]);
            std::vector<float> rows;
            if (!jlens_embed_rows(model, backend, buft, rid, rows, e2)) die(e2.c_str());
            const double F0 = later(ll0, s);
            std::vector<double> dl, ds, pr;
            for (int r = 0; r < K; ++r) {
                const int32_t c = rid[1 + r];
                const double p = br.prob[(size_t) s * K + r];
                if (c == act) { dl.push_back(0.0); ds.push_back(0.0); pr.push_back(p); continue; }
                std::vector<float> d(n_embd);
                double lin = 0.0;
                for (int64_t e = 0; e < n_embd; ++e) {
                    d[e] = rows[(size_t) (1 + r) * n_embd + e] - rows[e];
                    lin += (double) gemb[(size_t) (s + 1) * n_embd + e] * d[e];
                }
                double Fpm[2];
                for (int sg = 0; sg < 2; ++sg) {
                    for (int64_t e = 0; e < n_embd; ++e) add[(size_t) (s + 1) * n_embd + e] = (sg == 0 ? 1.0f : -1.0f) * fd_eps * d[e];
                    std::vector<double> llp;
                    if (!jlens_text_loglik(model, backend, buft, bt, chunk, n_real, add.data(), llp, e2)) die(e2.c_str());
                    if (pause_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms));
                    Fpm[sg] = later(llp, s);
                }
                std::fill(add.begin() + (size_t) (s + 1) * n_embd, add.begin() + (size_t) (s + 2) * n_embd, 0.0f);
                const double fd = (Fpm[0] - Fpm[1]) / (2.0 * fd_eps);
                std::vector<int32_t> sw = bt;
                sw[s + 1] = c;
                std::vector<double> llc;
                if (!jlens_text_loglik(model, backend, buft, sw, chunk, n_real, nullptr, llc, e2)) die(e2.c_str());
                if (pause_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms));
                const double swap = later(llc, s) - F0;
                double dn = 0.0, en = 0.0;
                for (int64_t e = 0; e < n_embd; ++e) { dn += (double) d[e] * d[e]; en += (double) rows[e] * rows[e]; }
                if (v_lin.size() < 14) fprintf(stderr, "    s=%3d c=%6d p=%.3f |d|=%.3f |E|=%.3f: linear %+.4f  fd %+.4f  swap %+.4f\n", s, c, p, std::sqrt(dn), std::sqrt(en), lin, fd, swap);
                v_lin.push_back(lin); v_fd.push_back(fd); v_swap.push_back(swap); v_dlin.push_back(br.delta[(size_t) s * K + r]);
                dl.push_back(lin); ds.push_back(swap); pr.push_back(p);
            }
            auto spread = [&](const std::vector<double> & dv) {
                double sp = 0, sd = 0, sdd = 0;
                for (size_t i = 0; i < dv.size(); ++i) { sp += pr[i]; sd += pr[i] * dv[i]; sdd += pr[i] * dv[i] * dv[i]; }
                const double mean = sd / sp;
                return std::sqrt(std::max(0.0, sdd / sp - mean * mean));
            };
            sp_lin.push_back(spread(dl)); sp_swap.push_back(spread(ds));
            fprintf(stderr, "  s=%3d spread linear %.3f swap %.3f (driver %.3f)\n", s, sp_lin.back(), sp_swap.back(), br.spread[s]);
        }
        auto pearson = [](const std::vector<double> & x, const std::vector<double> & y) {
            const size_t n = x.size();
            double mx = 0, my = 0;
            for (size_t i = 0; i < n; ++i) { mx += x[i]; my += y[i]; }
            mx /= n; my /= n;
            double sxy = 0, sxx = 0, syy = 0;
            for (size_t i = 0; i < n; ++i) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); syy += (y[i] - my) * (y[i] - my); }
            return sxy / std::sqrt(sxx * syy);
        };
        auto ranks = [](const std::vector<double> & x) {
            std::vector<size_t> o(x.size());
            for (size_t i = 0; i < o.size(); ++i) o[i] = i;
            std::sort(o.begin(), o.end(), [&](size_t a2, size_t b2) { return x[a2] < x[b2]; });
            std::vector<double> r(x.size());
            for (size_t i = 0; i < o.size(); ++i) r[o[i]] = (double) i;
            return r;
        };
        std::vector<double> rel;
        for (size_t i = 0; i < v_lin.size(); ++i) rel.push_back(std::fabs(v_fd[i] - v_lin[i]) / std::max(1e-9, std::fabs(v_lin[i])));
        std::vector<double> rs = rel;
        std::sort(rs.begin(), rs.end());
        double maxdrv = 0.0;
        for (size_t i = 0; i < v_lin.size(); ++i) maxdrv = std::max(maxdrv, std::fabs(v_dlin[i] - v_lin[i]) / std::max(1e-9, std::fabs(v_lin[i])));
        fprintf(stderr, "gradient vs finite difference (eps %.3f, %zu pairs): rel err median %.2e, p90 %.2e; corr %.4f; driver delta vs host dot max rel %.1e\n",
                fd_eps, rel.size(), rs[rs.size() / 2], rs[rs.size() * 9 / 10], pearson(v_lin, v_fd), maxdrv);
        fprintf(stderr, "linear vs actual swap (%zu pairs): pearson %.3f, spearman %.3f\n", v_lin.size(), pearson(v_lin, v_swap), pearson(ranks(v_lin), ranks(v_swap)));
        fprintf(stderr, "spread linear vs swap (%zu positions): pearson %.3f, spearman %.3f\n", sp_lin.size(), pearson(sp_lin, sp_swap), pearson(ranks(sp_lin), ranks(sp_swap)));
        return 0;
    }

    // ---- forward check: chunked forward against one unchunked forward (both explicit attention), on N tokens
    if (fwd_check > 0) {
        std::vector<int32_t> ft(ids.begin() + offset, ids.begin() + offset + fwd_check);
        jlens_value_result rf, rx;
        std::string e2;
        if (!jlens_value_exact(model, backend, buft, ft, L, chunk, true, rx, e2, /*full_fwd=*/true)) die(e2.c_str());
        if (!jlens_value_exact(model, backend, buft, ft, L, chunk, true, rf, e2, /*full_fwd=*/false)) die(e2.c_str());
        {
            std::vector<double> rel;
            for (int s = 0; s < fwd_check; ++s) {
                double d2 = 0.0, a2 = 0.0;
                for (int64_t e = 0; e < n_embd; ++e) {
                    const double x = rx.h[(size_t) s * n_embd + e], y = rf.h[(size_t) s * n_embd + e];
                    d2 += (x - y) * (x - y); a2 += x * x;
                }
                rel.push_back(std::sqrt(d2 / std::max(a2, 1e-30)));
            }
            std::vector<double> rs = rel;
            std::sort(rs.begin(), rs.end());
            fprintf(stderr, "chunked vs full forward, slot state h (%d tokens, chunk %d): rel err median %.2e, max %.2e; first chunk median %.2e\n",
                    fwd_check, rf.chunk, rs[rs.size() / 2], rs.back(), [&] { std::vector<double> f(rel.begin(), rel.begin() + std::min(fwd_check, rf.chunk)); std::sort(f.begin(), f.end()); return f[f.size() / 2]; }());
        }
        for (int kind = 0; kind < 2; ++kind) {
            const std::vector<float> & a = kind ? rx.now : rx.fut, & b = kind ? rf.now : rf.fut;
            std::vector<double> rel, lr;
            for (int s = 0; s + 2 < fwd_check; ++s) {
                double d2 = 0.0, a2 = 0.0;
                for (int64_t e = 0; e < n_embd; ++e) {
                    const double x = a[(size_t) s * n_embd + e], y = b[(size_t) s * n_embd + e];
                    d2 += (x - y) * (x - y); a2 += x * x;
                }
                rel.push_back(std::sqrt(d2 / std::max(a2, 1e-30)));
                const float na = kind ? rx.now_norm[s] : rx.fut_norm[s], nb = kind ? rf.now_norm[s] : rf.fut_norm[s];
                lr.push_back(std::fabs(std::log(nb / na)));
            }
            std::sort(rel.begin(), rel.end());
            std::sort(lr.begin(), lr.end());
            fprintf(stderr, "chunked vs full forward, %s (%d tokens): vector rel err median %.2e, p95 %.2e; |log norm ratio| median %.2e, p95 %.2e, max %.2e\n",
                    kind ? "now" : "fut", fwd_check, rel[rel.size() / 2], rel[rel.size() * 95 / 100], lr[lr.size() / 2], lr[lr.size() * 95 / 100], lr.back());
        }
        return 0;
    }

    // ---- exact
    jlens_value_result ex;
    std::string err;
    if (!jlens_value_exact(model, backend, buft, tk, L, chunk, /*keep_vectors=*/true, ex, err)) die(err.c_str());
    fprintf(stderr, "exact: T %d, chunk %d: forward %.2f s, main sweep %.2f s, now sweep %.2f s\n",
            T, ex.chunk, ex.sec_fwd, ex.sec_main, ex.sec_now);

    // ---- brute force (as homeo-value)
    std::vector<double> bf_fut((size_t) T * n_embd, 0.0), bf_now((size_t) T * n_embd, 0.0);
    {
        ggml_init_params sparams = { 8 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * ictx = ggml_init(sparams);
        ggml_tensor * seed_t   = ggml_new_tensor_1d(ictx, GGML_TYPE_F32, 1);
        ggml_tensor * t_tokens = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, T);
        ggml_tensor * t_pos    = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, T);
        ggml_tensor * t_mask   = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, T, T);
        ggml_tensor * t_labels = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, V, T);
        ggml_tensor * t_steer  = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, n_embd, T);
        ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors_from_buft(ictx, buft);
        if (!ibuf) die("input alloc failed");
        ggml_backend_buffer_clear(ibuf, 0);
        const float one = 1.0f;
        ggml_backend_tensor_set(seed_t, &one, 0, sizeof(float));
        std::vector<int32_t> posv(T);
        for (int i = 0; i < T; ++i) posv[i] = i;
        std::vector<float> mask((size_t) T * T);
        for (int j = 0; j < T; ++j)
            for (int i = 0; i < T; ++i)
                mask[i + (size_t) j * T] = (i <= j) ? 0.0f : -FLT_MAX;
        ggml_backend_tensor_set(t_tokens, tk.data(),   0, T * sizeof(int32_t));
        ggml_backend_tensor_set(t_pos,    posv.data(), 0, T * sizeof(int32_t));
        ggml_backend_tensor_set(t_mask,   mask.data(), 0, mask.size() * sizeof(float));

        jlens_steer_forward fwd;
        if (!jlens_build_forward_steer(model, T, L, fwd, /*per_position=*/true)) die("graph build failed");
        std::vector<ggml_tensor *> grad_accs(ggml_graph_n_nodes(fwd.gf), nullptr);
        for (int i = 0; i < ggml_graph_n_nodes(fwd.gf); ++i) {
            if (ggml_graph_node(fwd.gf, i) == fwd.loss) grad_accs[i] = seed_t;
        }
        ggml_cgraph * gb = ggml_graph_dup(fwd.gctx, fwd.gf, /*force_grads=*/true);
        ggml_build_backward_expand(fwd.gctx, gb, grad_accs.data());
        ggml_tensor * g_slot = ggml_graph_get_grad(gb, fwd.steer);
        if (!g_slot) die("no slot gradient");
        ggml_set_output(g_slot);
        auto wire = [](ggml_tensor * dst, const ggml_tensor * src) { dst->data = src->data; dst->buffer = src->buffer; };
        wire(fwd.tokens, t_tokens); wire(fwd.pos, t_pos); wire(fwd.mask, t_mask);
        wire(fwd.labels, t_labels); wire(fwd.steer, t_steer);
        ggml_gallocr_t galloc = ggml_gallocr_new(buft);
        if (!ggml_gallocr_alloc_graph(galloc, gb)) die("galloc failed");
        auto compute = [&]() { if (ggml_backend_graph_compute(backend, gb) != GGML_STATUS_SUCCESS) die("compute failed"); };

        // every row carries the model's own distribution (zero gradient) except the one-hot target row,
        // so d l_t / d h = -T * grad (cross_entropy_loss is the row mean)
        compute();
        std::vector<float> logits((size_t) V * T), plab((size_t) V * T), onehot(V, 0.0f);
        ggml_backend_tensor_get(fwd.logits, logits.data(), 0, logits.size() * sizeof(float));
        for (int t = 0; t < T; ++t) {
            const float * z = logits.data() + (size_t) t * V;
            const float mx = *std::max_element(z, z + V);
            double s = 0.0;
            for (int x = 0; x < V; ++x) s += std::exp((double) z[x] - mx);
            for (int x = 0; x < V; ++x) plab[(size_t) t * V + x] = (float) (std::exp((double) z[x] - mx) / s);
        }
        ggml_backend_tensor_set(t_labels, plab.data(), 0, plab.size() * sizeof(float));
        std::vector<float> g((size_t) T * n_embd);
        for (int t = 0; t + 1 < T; ++t) {
            onehot[tk[t + 1]] = 1.0f;
            ggml_backend_tensor_set(t_labels, onehot.data(), (size_t) t * V * sizeof(float), V * sizeof(float));
            onehot[tk[t + 1]] = 0.0f;
            compute();
            ggml_backend_tensor_set(t_labels, plab.data() + (size_t) t * V, (size_t) t * V * sizeof(float), V * sizeof(float));
            ggml_backend_tensor_get(g_slot, g.data(), 0, (size_t) (t + 1) * n_embd * sizeof(float));
            for (int s = 0; s <= t; ++s) {
                double * dst = (s == t ? bf_now.data() : bf_fut.data()) + (size_t) s * n_embd;
                for (int64_t e = 0; e < n_embd; ++e) dst[e] += -(double) T * g[(size_t) s * n_embd + e];
            }
        }
        ggml_gallocr_free(galloc);
        ggml_free(fwd.gctx);
        ggml_backend_buffer_free(ibuf);
        ggml_free(ictx);
    }

    // ---- compare: relative error and cosine per position
    auto cmp = [&](const char * name, const std::vector<double> & bf, const std::vector<float> & exv, int s_max) {
        std::vector<double> rel, cosv;
        double max_rel = 0.0;
        int worst = -1;
        for (int s = 0; s < s_max; ++s) {
            double d2 = 0.0, b2 = 0.0, e2 = 0.0, be = 0.0;
            for (int64_t e = 0; e < n_embd; ++e) {
                const double b = bf[(size_t) s * n_embd + e], x = exv[(size_t) s * n_embd + e];
                d2 += (b - x) * (b - x); b2 += b * b; e2 += x * x; be += b * x;
            }
            const double r = std::sqrt(d2 / std::max(b2, 1e-30));
            rel.push_back(r);
            cosv.push_back(be / std::sqrt(std::max(b2 * e2, 1e-60)));
            if (r > max_rel) { max_rel = r; worst = s; }
        }
        std::vector<double> rs = rel, cs = cosv;
        std::sort(rs.begin(), rs.end());
        std::sort(cs.begin(), cs.end());
        fprintf(stderr, "%s: %d positions, rel err median %.2e, max %.2e (s=%d); cosine min %.6f, median %.6f\n",
                name, s_max, rs[rs.size() / 2], max_rel, worst, cs.front(), cs[cs.size() / 2]);
    };
    {
        std::vector<double> bf_incl(bf_fut.size());
        std::vector<float> ex_incl(ex.fut.size());
        for (size_t i = 0; i < bf_incl.size(); ++i) { bf_incl[i] = bf_fut[i] + bf_now[i]; ex_incl[i] = ex.fut[i] + ex.now[i]; }
        cmp("incl", bf_incl, ex_incl, T - 1);
    }
    cmp("now", bf_now, ex.now, T - 1);
    cmp("fut", bf_fut, ex.fut, T - 2); // s = T-2 has no later target
    for (int s : { 0, 1, T / 4, T / 2, T - 8, T - 3 }) {
        double nb = 0.0, nn = 0.0;
        for (int64_t e = 0; e < n_embd; ++e) {
            nb += bf_fut[(size_t) s * n_embd + e] * bf_fut[(size_t) s * n_embd + e];
            nn += bf_now[(size_t) s * n_embd + e] * bf_now[(size_t) s * n_embd + e];
        }
        fprintf(stderr, "  s=%3d |fut| brute %.4f exact %.4f   |now| brute %.4f exact %.4f\n",
                s, std::sqrt(nb), ex.fut_norm[s], std::sqrt(nn), ex.now_norm[s]);
    }

    // ---- bench
    if (bench > 0) {
        std::vector<int32_t> bt(ids.begin() + offset, ids.begin() + offset + bench);
        jlens_value_result br;
        if (!jlens_value_exact(model, backend, buft, bt, L, bench_chunk, /*keep_vectors=*/false, br, err)) die(err.c_str());
        fprintf(stderr, "bench: %d tokens, chunk %d: forward %.2f s, main sweep %.2f s, now sweep %.2f s, total %.2f s\n",
                bench, br.chunk, br.sec_fwd, br.sec_main, br.sec_now, br.sec_fwd + br.sec_main + br.sec_now);
        const int nb = 16;
        for (int k = 0; k < nb; ++k) {
            const int s0 = k * bench / nb, s1 = (k + 1) * bench / nb;
            double f = 0.0, n = 0.0;
            for (int s = s0; s < s1; ++s) { f += std::log(std::max(br.fut_norm[s], 1e-12f)); n += std::log(std::max(br.now_norm[s], 1e-12f)); }
            fprintf(stderr, "  positions %5d-%5d (%5d later): mean log|fut| %+.3f, mean log|now| %+.3f\n",
                    s0, s1 - 1, bench - s1, f / (s1 - s0), n / (s1 - s0));
        }
    }
    return 0;
}
