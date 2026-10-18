// homeo-value: the value lens (the Value Lens proposal; NOTES-homeostasis.md).
//
// Prices residual-stream directions by their effect on the model's own future predictive success.
// Teacher-forced windows of real text (a raw i32 token-id stream). For every target position t, one
// backward pass of l_t = log p(x_{t+1} | x_<=t) gives d l_t / d h_s for all s <= t, at the slot layer
// (--layer; a zero per-position steering slot) and at every --read layer above it. Accumulated per s:
//   now   = d l_s / d h_s                      (k = 0: the J-lens view, for the actual token)
//   short = sum_{k=1..--short} d l_{s+k} / d h_s
//   fut   = sum_{k=1..--horizon} d l_{s+k} / d h_s
// The price of a direction u at s is u . g (nats of log-likelihood per unit push). Only positions with
// a full horizon inside the window are written.
//
// Output, one pair per window (an existing .json is skipped, so runs resume):
//   DIR/wNNNNN.bin  f16 [n_layers][3 (now, short, fut)][P][n_embd]
//   DIR/wNNNNN.json offset, tokens, per-position log-likelihood and entropy, layers, valid range,
//                   and the lag profile (mean |d l_{s+k} / d h_s| at the slot layer, k = 0..horizon)
//
// --fd-check N: at N positions of the first window, compare now / short / fut predicted by the
// gradients with central finite differences along random unit directions, then exit.
//
// usage: homeo-value -m model.gguf --ids ids.bin --out DIR [--n-windows 600] [--n-tokens 192]
//          [--skip 32] [--horizon 32] [--short 8] [--layer 40] [--read 44,50,56] [--seed 0]
//          [--fd-check N] [--fd-eps 8,16,32] [--stride N]
// --stride N: window w starts at token w * N instead of a random offset (for prepared streams of
// back-to-back windows, e.g. scenario sets).
// --pause-ms N: sleep N ms after each forward+backward pass (lower average GPU power, quieter fans). The
// value is re-read from DIR/pause_ms at the start of every window if that file exists, so it can be
// changed while the run is going: echo 400 > DIR/pause_ms

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
#include <sys/stat.h>
#include <unistd.h>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

static void die(const char * msg) {
    fprintf(stderr, "homeo-value: %s\n", msg);
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

static std::vector<int> parse_ints(const std::string & s) {
    std::vector<int> v;
    char * c = strdup(s.c_str());
    for (char * t = strtok(c, ","); t; t = strtok(nullptr, ",")) v.push_back(atoi(t));
    free(c);
    return v;
}

static bool file_exists(const std::string & p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

static long long file_size(const std::string & p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 ? (long long) st.st_size : -1;
}

// write then fsync then rename, so a crash leaves either the old state or the complete file
static void write_atomic(const std::string & path, const void * data, size_t n) {
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f || fwrite(data, 1, n, f) != n) die(("cannot write " + path).c_str());
    fflush(f);
    fsync(fileno(f));
    fclose(f);
    if (rename(tmp.c_str(), path.c_str()) != 0) die(("cannot rename " + tmp).c_str());
}

int main(int argc, char ** argv) {
    std::string model_path, ids_path, out_dir, read_str = "44,50,56", eps_str = "8,16,32";
    int n_windows = 600, T = 192, skip = 32, K = 32, K2 = 8, L = 40, seed = 0, fd_check = 0, stride = 0, pause_ms = 0;
    int n_threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) die(("missing value for " + a).c_str()); return argv[++i]; };
        if      (a == "-m")          model_path = next();
        else if (a == "--ids")       ids_path   = next();
        else if (a == "--out")       out_dir    = next();
        else if (a == "--n-windows") n_windows  = atoi(next());
        else if (a == "--n-tokens")  T          = atoi(next());
        else if (a == "--skip")      skip       = atoi(next());
        else if (a == "--horizon")   K          = atoi(next());
        else if (a == "--short")     K2         = atoi(next());
        else if (a == "--layer")     L          = atoi(next());
        else if (a == "--read")      read_str   = next();
        else if (a == "--seed")      seed       = atoi(next());
        else if (a == "--fd-check")  fd_check   = atoi(next());
        else if (a == "--fd-eps")    eps_str    = next();
        else if (a == "--threads")   n_threads  = atoi(next());
        else if (a == "--stride")    stride     = atoi(next());
        else if (a == "--pause-ms")  pause_ms   = atoi(next());
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || ids_path.empty() || (out_dir.empty() && fd_check == 0)) {
        fprintf(stderr, "usage: homeo-value -m model.gguf --ids ids.bin --out DIR [options]\n");
        return 1;
    }
    const int s_lo = skip, s_hi = T - 2 - K; // positions with a full horizon: [s_lo, s_hi]
    if (s_hi < s_lo || K2 > K) die("window too short for --skip/--horizon, or --short > --horizon");
    const int P = s_hi - s_lo + 1;

    // ---- token stream and window offsets ----
    std::vector<int32_t> ids;
    {
        std::ifstream f(ids_path, std::ios::binary | std::ios::ate);
        if (!f) die("cannot open --ids");
        const size_t n = (size_t) f.tellg() / sizeof(int32_t);
        ids.resize(n);
        f.seekg(0);
        f.read((char *) ids.data(), n * sizeof(int32_t));
    }
    if ((int) ids.size() < T + 1) die("token stream shorter than one window");
    std::vector<size_t> offsets(n_windows);
    if (stride > 0) {
        if ((size_t) (n_windows - 1) * stride + T > ids.size()) die("--stride: stream too short for --n-windows");
        for (int w = 0; w < n_windows; ++w) offsets[w] = (size_t) w * stride;
    } else {
        std::mt19937_64 rng(seed);
        for (auto & o : offsets) o = (size_t) (rng() % (uint64_t) (ids.size() - T));
    }

    // ---- model + graph ----
    llama_backend_init();
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(0);
    if (!backend) die("cuda init failed");
    jlens_model model;
    if (!jlens_model_load(model_path.c_str(), buft, model)) die("model load failed");
    const int64_t n_embd = model.hparams.n_embd;
    const int n_layer = (int) model.hparams.n_layer;
    if (L < 0 || L >= n_layer - 1) die("--layer out of range");
    std::vector<int> layers = { L };
    for (int l : parse_ints(read_str)) {
        if (l <= L || l >= n_layer) die("--read layers must be above --layer and below n_layer");
        layers.push_back(l);
    }
    const int NL = (int) layers.size();
    const int V = (int) model.get("output.weight")->ne[1];

    ggml_init_params sparams = { 4 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
    ggml_context * sctx = ggml_init(sparams);
    ggml_tensor * seed_t = ggml_new_tensor_1d(sctx, GGML_TYPE_F32, 1);
    if (!ggml_backend_alloc_ctx_tensors_from_buft(sctx, buft)) die("seed alloc failed");
    const float one = 1.0f;
    ggml_backend_tensor_set(seed_t, &one, 0, sizeof(float));

    // inputs get storage outside gallocr (as in homeo-steeropt)
    ggml_init_params inparams = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * inctx = ggml_init(inparams);
    ggml_tensor * t_tokens = ggml_new_tensor_1d(inctx, GGML_TYPE_I32, T);
    ggml_tensor * t_pos    = ggml_new_tensor_1d(inctx, GGML_TYPE_I32, T);
    ggml_tensor * t_mask   = ggml_new_tensor_2d(inctx, GGML_TYPE_F32, T, T);
    ggml_tensor * t_labels = ggml_new_tensor_2d(inctx, GGML_TYPE_F32, V, T);
    ggml_tensor * t_steer  = ggml_new_tensor_2d(inctx, GGML_TYPE_F32, n_embd, T);
    if (!ggml_backend_alloc_ctx_tensors_from_buft(inctx, buft)) die("input alloc failed");
    {
        std::vector<int32_t> posv(T);
        for (int i = 0; i < T; ++i) posv[i] = i;
        std::vector<float> mask((size_t) T * T);
        for (int j = 0; j < T; ++j)
            for (int i = 0; i < T; ++i)
                mask[i + (size_t) j * T] = (i <= j) ? 0.0f : -FLT_MAX;
        ggml_backend_tensor_set(t_pos,  posv.data(), 0, posv.size() * sizeof(int32_t));
        ggml_backend_tensor_set(t_mask, mask.data(), 0, mask.size() * sizeof(float));
        std::vector<float> z((size_t) V * T, 0.0f);
        ggml_backend_tensor_set(t_labels, z.data(), 0, z.size() * sizeof(float));
        std::vector<float> zs((size_t) n_embd * T, 0.0f);
        ggml_backend_tensor_set(t_steer, zs.data(), 0, zs.size() * sizeof(float));
    }

    jlens_steer_forward fwd;
    if (!jlens_build_forward_steer(model, T, L, fwd, /*per_position=*/true)) die("graph build failed");
    std::vector<ggml_tensor *> grad_accs(ggml_graph_n_nodes(fwd.gf), nullptr);
    {
        bool found = false;
        for (int i = 0; i < ggml_graph_n_nodes(fwd.gf); ++i) {
            if (ggml_graph_node(fwd.gf, i) == fwd.loss) { grad_accs[i] = seed_t; found = true; break; }
        }
        if (!found) die("loss node not found");
    }
    ggml_cgraph * gb = ggml_graph_dup(fwd.gctx, fwd.gf, /*force_grads=*/true);
    ggml_build_backward_expand(fwd.gctx, gb, grad_accs.data());
    std::vector<ggml_tensor *> grads(NL, nullptr);
    grads[0] = ggml_graph_get_grad(gb, fwd.steer); // slot layer: d loss / d h_L at every position
    for (int li = 1; li < NL; ++li) grads[li] = ggml_graph_get_grad(gb, fwd.l_out[layers[li]]);
    for (int li = 0; li < NL; ++li) {
        if (!grads[li]) die("missing gradient for a read layer");
        ggml_set_output(grads[li]);
    }
    {
        auto wire = [](ggml_tensor * dst, const ggml_tensor * src) { dst->data = src->data; dst->buffer = src->buffer; };
        wire(fwd.tokens, t_tokens); wire(fwd.pos, t_pos); wire(fwd.mask, t_mask);
        wire(fwd.labels, t_labels); wire(fwd.steer, t_steer);
    }
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(galloc, gb)) die("galloc failed");
    fprintf(stderr, "homeo-value: T %d, slot layer %d, %d layers read, compute buffer %.2f GiB, %d positions per window\n",
            T, L, NL, ggml_gallocr_get_buffer_size(galloc, 0) / (1024.0 * 1024.0 * 1024.0), P);

    // ggml_cross_entropy_loss is the mean over all T rows, and its backward is (softmax(z_r) - y_r) / T for
    // every row r: it assumes each label row sums to 1, so an all-zero row would leak softmax(z_r). Every row
    // therefore carries the model's own distribution as its label (zero gradient), and only row t carries
    // the one-hot next token. The loss gradient is then -(1/T) d l_t / d h, so d l_t / d h = -T * grad.
    const float gscale = -(float) T;
    std::vector<float> logits((size_t) V * T), plab((size_t) V * T), onehot(V, 0.0f);
    std::vector<float> gbuf((size_t) (K + 1) * n_embd);

    auto compute = [&]() {
        if (ggml_backend_graph_compute(backend, gb) != GGML_STATUS_SUCCESS) die("compute failed");
    };
    auto set_onehot = [&](int t, int tok) {
        onehot[tok] = 1.0f;
        ggml_backend_tensor_set(t_labels, onehot.data(), (size_t) t * V * sizeof(float), V * sizeof(float));
        onehot[tok] = 0.0f;
    };
    auto restore_row = [&](int t) {
        ggml_backend_tensor_set(t_labels, plab.data() + (size_t) t * V, (size_t) t * V * sizeof(float), V * sizeof(float));
    };
    // from the current logits: per-position log-likelihood of the next token and entropy, and (with
    // set_self) the model's own distribution uploaded as every row's label
    auto read_logits = [&](const std::vector<int32_t> & tk, std::vector<double> & ll, std::vector<double> & H, bool set_self) {
        ggml_backend_tensor_get(fwd.logits, logits.data(), 0, logits.size() * sizeof(float));
        ll.assign(T, 0.0); H.assign(T, 0.0);
        parallel_for(T, n_threads, [&](int t) {
            const float * z = logits.data() + (size_t) t * V;
            float * pr = set_self ? plab.data() + (size_t) t * V : nullptr; // steered passes must not touch the labels
            float m = -INFINITY;
            for (int x = 0; x < V; ++x) m = std::max(m, z[x]);
            double s = 0.0;
            for (int x = 0; x < V; ++x) s += std::exp((double) z[x] - m);
            const double lse = m + std::log(s);
            double h = 0.0;
            for (int x = 0; x < V; ++x) {
                const double lp = z[x] - lse, p = std::exp(lp);
                h -= p * lp;
                if (pr) pr[x] = (float) p;
            }
            if (t < T - 1) ll[t] = z[tk[t + 1]] - lse;
            H[t] = h;
        });
        if (set_self) ggml_backend_tensor_set(t_labels, plab.data(), 0, plab.size() * sizeof(float));
    };
    // backward of l_t alone; returns d l_t / d h_s for s in [lo, t] at layer index li in gbuf
    auto read_grads = [&](int li, int lo, int t) {
        ggml_backend_tensor_get(grads[li], gbuf.data(), (size_t) lo * n_embd * sizeof(float),
                                (size_t) (t - lo + 1) * n_embd * sizeof(float));
        for (size_t i = 0; i < (size_t) (t - lo + 1) * n_embd; ++i) gbuf[i] *= gscale;
    };

    // ---------------------------------------------------------------- finite-difference check
    if (fd_check > 0) {
        std::vector<int32_t> tk(ids.begin() + offsets[0], ids.begin() + offsets[0] + T);
        ggml_backend_tensor_set(t_tokens, tk.data(), 0, T * sizeof(int32_t));
        {
            std::vector<double> ll0, H0;
            compute();
            read_logits(tk, ll0, H0, /*set_self=*/true);
        }
        std::mt19937_64 rng(seed + 11);
        std::normal_distribution<float> nd;
        const std::vector<int> epsv = parse_ints(eps_str);
        for (int c = 0; c < fd_check; ++c) {
            const int s = s_lo + (int) ((uint64_t) (c + 1) * (uint64_t) P / (uint64_t) (fd_check + 1));
            // analytic: one backward per t in [s, s + K], slot layer, column s -> g_now, g_short, g_fut
            std::vector<double> gn(n_embd, 0.0), gs(n_embd, 0.0), gf(n_embd, 0.0);
            for (int t = s; t <= s + K; ++t) {
                set_onehot(t, tk[t + 1]);
                compute();
                restore_row(t);
                read_grads(0, s, s);
                for (int e = 0; e < n_embd; ++e) {
                    if (t == s) gn[e] = gbuf[e];
                    else { gf[e] += gbuf[e]; if (t - s <= K2) gs[e] += gbuf[e]; }
                }
            }
            // directions: along g_now, along g_fut (largest predicted effects), and one random direction
            std::vector<std::vector<float>> dirs(3, std::vector<float>(n_embd));
            const char * dname[3] = { "g_now", "g_fut", "random" };
            for (int di = 0; di < 3; ++di) {
                double n2 = 0.0;
                for (int e = 0; e < n_embd; ++e) {
                    const double x = di == 0 ? gn[e] : di == 1 ? gf[e] : (double) nd(rng);
                    dirs[di][e] = (float) x; n2 += x * x;
                }
                for (auto & x : dirs[di]) x /= (float) std::sqrt(n2);
            }
            for (int di = 0; di < 3; ++di) {
                const std::vector<float> & u = dirs[di];
                double a_now = 0.0, a_short = 0.0, a_fut = 0.0;
                for (int e = 0; e < n_embd; ++e) { a_now += u[e] * gn[e]; a_short += u[e] * gs[e]; a_fut += u[e] * gf[e]; }
                for (int eps : epsv) {
                    std::vector<double> llp, llm, H;
                    std::vector<float> col(n_embd);
                    for (int sign : { +1, -1 }) {
                        for (int e = 0; e < n_embd; ++e) col[e] = sign * eps * u[e];
                        ggml_backend_tensor_set(t_steer, col.data(), (size_t) s * n_embd * sizeof(float), n_embd * sizeof(float));
                        compute();
                        read_logits(tk, sign > 0 ? llp : llm, H, /*set_self=*/false);
                    }
                    std::fill(col.begin(), col.end(), 0.0f);
                    ggml_backend_tensor_set(t_steer, col.data(), (size_t) s * n_embd * sizeof(float), n_embd * sizeof(float));
                    auto d = [&](int t) { return (llp[t] - llm[t]) / (2.0 * eps); };
                    double f_now = d(s), f_short = 0.0, f_fut = 0.0;
                    for (int k = 1; k <= K; ++k) { f_fut += d(s + k); if (k <= K2) f_short += d(s + k); }
                    fprintf(stderr, "fd s=%3d u=%-6s eps=%3d: now %+.3e / %+.3e (%6.3f)  short %+.3e / %+.3e (%6.3f)  fut %+.3e / %+.3e (%6.3f)\n",
                            s, dname[di], eps, f_now, a_now, f_now / a_now, f_short, a_short, f_short / a_short, f_fut, a_fut, f_fut / a_fut);
                }
            }
        }
        return 0;
    }

    // ---------------------------------------------------------------- windows
    mkdir(out_dir.c_str(), 0755);
    {
        json meta;
        meta["model"] = model_path; meta["ids"] = ids_path; meta["n_tokens"] = T; meta["skip"] = skip;
        meta["horizon"] = K; meta["short"] = K2; meta["layers"] = layers; meta["slot_layer"] = L;
        meta["positions"] = { s_lo, s_hi }; meta["n_embd"] = n_embd; meta["kinds"] = { "now", "short", "fut" };
        meta["layout"] = "f16 [n_layers][3][P][n_embd], P = positions[1] - positions[0] + 1";
        meta["seed"] = seed; meta["n_windows"] = n_windows; meta["stride"] = stride;
        std::ofstream(out_dir + "/meta.json") << meta.dump(2) << "\n";
    }
    std::vector<float> acc((size_t) NL * 3 * P * n_embd);
    std::vector<ggml_fp16_t> acc16(acc.size());
    const auto t_start = std::chrono::steady_clock::now();
    int done = 0;
    for (int w = 0; w < n_windows; ++w) {
        char base[64];
        snprintf(base, sizeof(base), "/w%05d", w);
        const std::string jpath = out_dir + base + ".json", bpath = out_dir + base + ".bin";
        const long long want = (long long) acc.size() * (long long) sizeof(ggml_fp16_t);
        if (file_exists(jpath) && file_size(bpath) == want) continue; // resume: both files present and complete
        {
            std::ifstream pf(out_dir + "/pause_ms");
            int v;
            if (pf >> v && v >= 0 && v != pause_ms) {
                pause_ms = v;
                fprintf(stderr, "homeo-value: pause %d ms per pass\n", pause_ms);
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<int32_t> tk(ids.begin() + offsets[w], ids.begin() + offsets[w] + T);
        ggml_backend_tensor_set(t_tokens, tk.data(), 0, T * sizeof(int32_t));
        std::fill(acc.begin(), acc.end(), 0.0f);
        std::vector<double> lag_sum(K + 1, 0.0);
        std::vector<int> lag_n(K + 1, 0);
        std::vector<double> ll, H;
        compute(); // forward for this window's logits (labels do not affect them)
        read_logits(tk, ll, H, /*set_self=*/true);
        for (int t = s_lo; t <= T - 2; ++t) {
            set_onehot(t, tk[t + 1]);
            compute();
            restore_row(t);
            if (pause_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms));
            const int lo = std::max(s_lo, t - K);
            for (int li = 0; li < NL; ++li) {
                read_grads(li, lo, t);
                for (int s = lo; s <= std::min(t, s_hi); ++s) {
                    const int k = t - s;
                    const float * g = gbuf.data() + (size_t) (s - lo) * n_embd;
                    const size_t p = (size_t) (s - s_lo);
                    if (k == 0) {
                        float * dst = acc.data() + (((size_t) li * 3 + 0) * P + p) * n_embd;
                        for (int e = 0; e < n_embd; ++e) dst[e] = g[e];
                    } else {
                        float * fut = acc.data() + (((size_t) li * 3 + 2) * P + p) * n_embd;
                        for (int e = 0; e < n_embd; ++e) fut[e] += g[e];
                        if (k <= K2) {
                            float * sh = acc.data() + (((size_t) li * 3 + 1) * P + p) * n_embd;
                            for (int e = 0; e < n_embd; ++e) sh[e] += g[e];
                        }
                    }
                }
                if (li == 0) {
                    for (int s = lo; s <= t; ++s) {
                        const float * g = gbuf.data() + (size_t) (s - lo) * n_embd;
                        double n2 = 0.0;
                        for (int e = 0; e < n_embd; ++e) n2 += (double) g[e] * g[e];
                        lag_sum[t - s] += std::sqrt(n2);
                        lag_n[t - s]++;
                    }
                }
            }
        }
        ggml_fp32_to_fp16_row(acc.data(), acc16.data(), (int64_t) acc.size());
        write_atomic(bpath, acc16.data(), acc16.size() * sizeof(ggml_fp16_t));
        json j;
        j["window"] = w; j["offset"] = offsets[w]; j["tokens"] = tk;
        j["ll"] = std::vector<double>(ll.begin(), ll.end() - 1); j["entropy"] = std::vector<double>(H.begin(), H.end() - 1);
        std::vector<double> lag(K + 1);
        for (int k = 0; k <= K; ++k) lag[k] = lag_n[k] ? lag_sum[k] / lag_n[k] : 0.0;
        j["lag_norm"] = lag;
        const std::string js = j.dump() + "\n";
        write_atomic(jpath, js.data(), js.size()); // after the .bin, so a present .json implies a complete .bin
        done++;
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double tot = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        fprintf(stderr, "homeo-value: window %d done in %.1f s (%d this run, %.2f h elapsed)\n", w, sec, done, tot / 3600.0);
    }
    fprintf(stderr, "homeo-value: finished %d windows\n", done);
    return 0;
}
