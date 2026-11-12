#include "server-steer.h"
#include "server-jlens.h"

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "log.h"
#include "src/llama-model.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>

namespace {

// copies l_out-L (the block output, push included) of every position in the batch
struct steer_capture {
    std::string        name;
    std::vector<float> buf;
    int64_t            rows    = 0;
    int64_t            n_embd  = 0;
    bool               enabled = false;
    bool               append  = false; // explore: a decode spans several ubatches; rows accumulate in batch order
    // guided sampling: per row, the sum over the layers in lin_on of w_L . h_L, and |h|^2 at those layers
    const float *       lin_w = nullptr; // [n_layer][n_embd]
    std::vector<char>   lin_on;          // per layer; empty: capture l_out-L named by name instead
    std::vector<double> lin_acc, lin_h2; // [rows]
};

bool guide_cb_eval(steer_capture * c, ggml_tensor * t, bool ask) {
    const int L = std::strncmp(t->name, "l_out-", 6) == 0 ? std::atoi(t->name + 6) : -1;
    if (L < 0 || L >= (int) c->lin_on.size() || !c->lin_on[L]) return !ask;
    if (ask) return true;
    const int64_t rows = ggml_nrows(t);
    if (t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd || !ggml_is_contiguous(t) || (c->rows != 0 && c->rows != rows)) {
        c->rows = -1;
        return true;
    }
    c->rows = rows;
    c->buf.resize((size_t) rows * c->n_embd);
    ggml_backend_tensor_get(t, c->buf.data(), 0, ggml_nbytes(t));
    c->lin_acc.resize(rows, 0.0);
    c->lin_h2.resize(rows, 0.0);
    const float * w = c->lin_w + (size_t) L * c->n_embd;
    for (int64_t r = 0; r < rows; ++r) {
        const float * h = c->buf.data() + (size_t) r * c->n_embd;
        double d = 0.0, s2 = 0.0;
        for (int64_t e = 0; e < c->n_embd; ++e) { d += (double) w[e] * h[e]; s2 += (double) h[e] * h[e]; }
        c->lin_acc[r] += d;
        c->lin_h2[r]  += s2;
    }
    return true;
}

bool steer_cb_eval(ggml_tensor * t, bool ask, void * ud) {
    auto * c = (steer_capture *) ud;
    if (!c->enabled) return !ask;
    if (c->rows < 0) return !ask; // capture failed: the caller sees rows < 0
    if (!c->lin_on.empty()) return guide_cb_eval(c, t, ask);
    if (c->name != t->name) return !ask;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd || !ggml_is_contiguous(t)) {
        c->rows = -1; // checked by the caller
        return true;
    }
    const int64_t off = c->append ? c->rows : 0;
    c->rows = off + ggml_nrows(t);
    c->buf.resize((size_t) c->rows * c->n_embd);
    ggml_backend_tensor_get(t, c->buf.data() + (size_t) off * c->n_embd, 0, ggml_nbytes(t));
    return true;
}

// exposure counts of a log group, written as one row
struct steer_group {
    std::string kind, name, mode, model, note; // kind: steer or guide
    long long   tok = 0;
    int         n_ctx = 0, n_end = 0, n_req = 0;
    float       max_dose = 0.0f, h_lo = 0.0f, h_hi = 0.0f, max_beta = 0.0f;
    std::vector<std::string> dirs;
    std::chrono::steady_clock::time_point t0;
};

struct steer_state {
    std::mutex      mutex;
    steer_group     grp;
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
    steer_capture   cap;
    std::string     model_name;
    std::string     log_path;
    float           max_dose = 1.0f;
};

steer_state g_steer;

constexpr int N_CTX     = 32768; // explore: the walkers' live tree
constexpr int N_BATCH   = 2048;  // explore: one row per walker in a step
constexpr int N_UBATCH  = 256;   // l_out capture sees one ubatch per callback: steer/guide/superpose decodes stay within it
constexpr int N_ROWS_1U = N_UBATCH - 1; // their sequence budget (explore's capture appends across ubatches instead)
constexpr int N_SEQ_MAX = 2048;  // LLAMA_MAX_SEQ (raised in this fork); steering: the last one holds the shared unsteered prefix
constexpr int SUP_K_MAX = 1024; // superpose: tokens in the input
constexpr int N_FOLLOW  = 256;

// humane endpoints, as homeo-ident
constexpr int   LOOP_MAX_P    = 16;
constexpr int   LOOP_MIN_SPAN = 12;
constexpr float INCOH_H       = 6.0f;
constexpr int   INCOH_RUN     = 16;

void log_softmax(const float * z, float * out, int n, float temp) {
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) m = std::max(m, z[i] / temp);
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::exp(z[i] / temp - m);
    const float lse = m + (float) std::log(s);
    for (int i = 0; i < n; ++i) out[i] = z[i] / temp - lse;
}

// smallest period p whose repeat covers the last max(3p, min_span) tokens; 0 if none
int find_loop(const std::vector<int32_t> & g, int max_p, int min_span) {
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

std::string csv_q(const std::string & s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"') o += '"'; o += c; }
    return o + "\"";
}

const char * ENDPOINT_DESC = "loop: period<=16 over max(3p;12) tokens | incoh: H>6.0 for 16 steps";

void write_log_row(const std::string & row) {
    if (!g_steer.log_path.empty()) {
        if (FILE * f = fopen(g_steer.log_path.c_str(), "ab")) { fputs(row.c_str(), f); fclose(f); return; }
        LOG_WRN("steer: cannot append to %s; exposure row: %s", g_steer.log_path.c_str(), row.c_str());
    } else {
        LOG_INF("steer: exposure row (no --steer-log): %s", row.c_str());
    }
}

std::string today() {
    char date[16];
    const std::time_t now = std::time(nullptr);
    std::strftime(date, sizeof(date), "%Y-%m-%d", std::localtime(&now));
    return date;
}

// caller holds g_steer.mutex
void flush_group() {
    steer_group & g = g_steer.grp;
    if (g.n_req == 0) { g = steer_group(); return; }
    const std::string names = g.dirs.size() <= 4 ? [&] { std::string o; for (auto & d : g.dirs) o += (o.empty() ? "" : " ") + d; return o; }() : "";
    char manip[768];
    if (g.kind == "guide") {
        snprintf(manip, sizeof(manip), "probe-guided sampling, no push: each token chosen among the top candidates by %zu metric(s) (%s); %s",
                 g.dirs.size(), names.empty() ? "several" : names.c_str(), g.mode == "maximize" ? "maximize" :
                 ("tilt p(x) exp(beta z), |beta| <= " + std::to_string(g.max_beta).substr(0, 5)).c_str());
    } else if (g.kind == "explore") {
        snprintf(manip, sizeof(manip), "exploration with weighted walkers, no push: ancestral sampling with unbiased resampling of the walkers; %s",
                 g.mode.c_str());
    } else if (g.kind == "superpose") {
        snprintf(manip, sizeof(manip), "superposed input, no push: at every step the probability-weighted mean of the top candidates' input "
                 "embeddings is fed instead of a token (off the token manifold); %s", g.mode.c_str());
    } else {
        snprintf(manip, sizeof(manip), "steer along %zu layer-sweep probe directions (%s) x dose x mean|h| (%.1f-%.1f); %s", g.dirs.size(),
                 names.empty() ? "blocks of each" : names.c_str(), g.h_lo, g.h_hi, g.mode.c_str());
    }
    char row[2048];
    snprintf(row, sizeof(row), "%s,%s,%s,%s,%.3f,%lld,%d,%s,%.3f,%s\n", today().c_str(), ("loom-" + g.kind).c_str(),
             csv_q(g.model).c_str(), csv_q(manip).c_str(),
             g.max_dose, g.tok, g.n_ctx, csv_q(ENDPOINT_DESC).c_str(), g.n_ctx ? (double) g.n_end / g.n_ctx : 0.0,
             csv_q("llama-server /steer" + std::string(g.kind == "steer" ? "" : " (" + g.kind + ")") + ", log group '" + g.name + "': " + std::to_string(g.n_req) + " requests" + (g.note.empty() ? "" : "; " + g.note)).c_str());
    write_log_row(row);
    g = steer_group();
}

template <typename F>
void parallel_for(int n, int nt, F && fn) {
    nt = std::max(1, std::min(n, nt));
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; ++t) ts.emplace_back([&, t] { for (int j = t; j < n; j += nt) fn(j); });
    for (auto & t : ts) t.join();
}

bool set_push(llama_context * ctx, const std::vector<float> & u, int layer, float push, int n_embd, int n_layer,
              std::vector<float> & cvec) {
    if (push == 0.0f) return llama_set_adapter_cvec(ctx, nullptr, 0, n_embd, 0, 0) == 0;
    cvec.assign((size_t) n_embd * n_layer, 0.0f);
    // cvec data starts at layer 1; the vector for layer L is added to the output of block L
    for (int e = 0; e < n_embd; ++e) cvec[(size_t) (layer - 1) * n_embd + e] = push * u[e];
    return llama_set_adapter_cvec(ctx, cvec.data(), cvec.size(), n_embd, layer, layer) == 0;
}

// the steering context, created on first use; caller holds g_steer.mutex
bool ensure_ctx(std::string & err) {
    if (g_steer.ctx) return true;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = N_CTX;
    cp.n_batch           = N_BATCH;
    cp.n_ubatch          = N_UBATCH;
    cp.n_seq_max         = N_SEQ_MAX;
    cp.kv_unified        = true;
    cp.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval           = steer_cb_eval;
    cp.cb_eval_user_data = &g_steer.cap;
    cp.n_threads         = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    cp.n_threads_batch   = cp.n_threads;
    cp.no_perf           = true;
    g_steer.ctx = llama_init_from_model(g_steer.model, cp);
    if (!g_steer.ctx) { err = "steer: context init failed (out of VRAM?)"; return false; }
    return true;
}

// inverse normal CDF (Acklam), as the loom's probit
double probit(double p) {
    static const double a[] = {-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239};
    static const double b[] = {-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572};
    static const double c[] = {-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783};
    static const double d[] = {0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416};
    auto tail = [&](double q) { return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); };
    if (p < 0.02425) return tail(std::sqrt(-2 * std::log(p)));
    if (p > 0.97575) return -tail(std::sqrt(-2 * std::log(1 - p)));
    const double q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// percentile of x on a color scale (linear between the stored quantiles, 0.5 / 99.5 outside), as the loom's pctOf
double scale_pct(const std::vector<float> & q, const std::vector<float> & v, double x) {
    if (x <= v.front()) return 0.5;
    if (x >= v.back()) return 99.5;
    for (size_t k = 1; k < q.size(); ++k) {
        if (x <= v[k]) return v[k] > v[k - 1] ? q[k - 1] + (q[k] - q[k - 1]) * (x - v[k - 1]) / (v[k] - v[k - 1]) : q[k];
    }
    return 99.5;
}

} // namespace

void server_steer_init(llama_model * model, const std::string & model_name, const std::string & log_path, float max_dose) {
    std::lock_guard<std::mutex> lock(g_steer.mutex);
    if (g_steer.ctx) { llama_free(g_steer.ctx); g_steer.ctx = nullptr; }
    g_steer.model      = model;
    g_steer.model_name = model_name;
    g_steer.log_path   = log_path;
    g_steer.max_dose   = max_dose;
}

void server_steer_deinit() {
    std::lock_guard<std::mutex> lock(g_steer.mutex);
    flush_group();
    if (g_steer.ctx) { llama_free(g_steer.ctx); g_steer.ctx = nullptr; }
    g_steer.model = nullptr;
}

bool server_steer_run(const llama_vocab * vocab, const server_steer_params & p, server_steer_result & out, std::string & err) {
    std::lock_guard<std::mutex> lock(g_steer.mutex);
    const auto t_start = std::chrono::steady_clock::now();
    if (!g_steer.model) { err = "steer: not initialized (model reloading?)"; return false; }

    int layer = -1;
    float h_norm = 0.0f;
    std::vector<float> u;
    if (!p.dir.empty()) {
        if (!server_jlens_dir_get(p.dir, layer, h_norm, u)) { err = "steer: unknown direction '" + p.dir + "' (start the server with --probe-dirs)"; return false; }
    } else {
        // a raw direction: normalized; the dose unit is the mean |h| of a probe direction at the same layer (else measured on the prefix)
        layer = p.layer;
        double n2 = 0.0;
        for (float x : p.vector) n2 += (double) x * x;
        if (n2 <= 0.0) { err = "steer: need \"dir\", or \"vector\" and \"layer\""; return false; }
        u.resize(p.vector.size());
        for (size_t e = 0; e < u.size(); ++e) u[e] = (float) (p.vector[e] / std::sqrt(n2));
        h_norm = server_jlens_layer_h_norm(layer);
    }
    if (layer < 1 || layer >= llama_model_n_layer(g_steer.model)) { err = "steer: layer must be in [1, n_layer)"; return false; }
    if (p.set && !p.impulse) { err = "steer: set mode needs mode \"impulse\""; return false; }

    const int n_embd  = llama_model_n_embd(g_steer.model);
    const int n_layer = llama_model_n_layer(g_steer.model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if ((int) u.size() != n_embd) { err = "steer: direction size does not match the model"; return false; }

    std::vector<float> doses = {0.0f};
    for (float d : (p.set ? std::vector<float>{1.0f} : p.doses)) {
        if (!std::isfinite(d) || std::fabs(d) > g_steer.max_dose) {
            err = "steer: |dose| must be <= " + std::to_string(g_steer.max_dose) + " (--steer-max-dose)";
            return false;
        }
        if (d != 0.0f) doses.push_back(d);
    }
    const int n_arm = (int) doses.size();
    const int NS = p.n_seq, G = p.n_gen;
    const int s = (int) p.prefix.size() - 1; // the push token
    const int n_f = std::min((int) p.follow.size(), N_FOLLOW);
    if (s < 0) { err = "steer: empty prefix"; return false; }
    if (NS < 1 || G < 1 || G > 512 || n_arm * NS > N_ROWS_1U) {
        err = "steer: need 1 <= n_gen <= 512 and (arms x n_seq) <= " + std::to_string(N_ROWS_1U);
        return false;
    }
    if (!(p.temp > 0.0f)) { err = "steer: temperature must be > 0"; return false; }
    if (s + n_arm * NS * (G + 1) > N_CTX || s + n_arm * (n_f + 1) > N_CTX) {
        err = "steer: text too long for the steering context (" + std::to_string(N_CTX) + " cells): shorten the text, n_seq or n_gen";
        return false;
    }

    if (!ensure_ctx(err)) return false;
    llama_context * ctx = g_steer.ctx;
    llama_memory_t  mem = llama_get_memory(ctx);
    steer_capture & cap = g_steer.cap;
    cap.name    = "l_out-" + std::to_string(layer);
    cap.n_embd  = n_embd;
    cap.enabled = false;
    const int n_threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    const int MS = N_SEQ_MAX - 1;

    std::vector<float> cvec;
    llama_memory_clear(mem, true);
    if (!set_push(ctx, u, layer, 0.0f, n_embd, n_layer, cvec)) { err = "steer: cvec clear failed"; return false; }
    llama_batch batch = llama_batch_init(std::max(N_BATCH, N_SEQ_MAX), 0, 1);
    struct batch_guard { llama_batch & b; ~batch_guard() { llama_batch_free(b); } } bg{batch};
    auto fail = [&](const std::string & m) {
        err = m;
        cap.enabled = false;
        set_push(ctx, u, layer, 0.0f, n_embd, n_layer, cvec);
        llama_memory_clear(mem, true);
        return false;
    };

    // ---- the shared unsteered prefix [0, s); also the mean |h_L| if the probe file has none ----
    double hn_sum = 0.0;
    int64_t hn_n = 0;
    for (int cur = 0; cur < s; cur += N_UBATCH) {
        const int n = std::min(N_UBATCH, s - cur);
        batch.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            batch.token[j] = p.prefix[cur + j]; batch.pos[j] = cur + j; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = MS;
            batch.logits[j] = 0;
        }
        cap.enabled = h_norm <= 0.0f;
        cap.rows = 0;
        if (llama_decode(ctx, batch) != 0) return fail("steer: decode failed (prefix)");
        cap.enabled = false;
        if (h_norm <= 0.0f) {
            if (cap.rows != n) return fail("steer: l_out capture failed (prefix)");
            for (int j = 0; j < n; ++j) {
                if (cur + j < 16 && s > 32) continue;
                double s2 = 0.0;
                const float * h = cap.buf.data() + (size_t) j * n_embd;
                for (int e = 0; e < n_embd; ++e) s2 += (double) h[e] * h[e];
                hn_sum += std::sqrt(s2);
                hn_n++;
            }
        }
    }
    if (h_norm <= 0.0f) {
        if (hn_n == 0) return fail("steer: the probe file has no h_norms and the prefix is too short to measure |h|");
        h_norm = (float) (hn_sum / hn_n);
    }
    out.layer  = layer;
    out.h_norm = h_norm;
    out.arms.assign(n_arm, server_steer_arm());
    float set_push_val = 0.0f;
    if (p.set) {
        // the push token unpushed, to read its u.h and |h|; then push = set_cos |h| - u.h
        llama_memory_seq_rm(mem, 0, -1, -1);
        if (s > 0) llama_memory_seq_cp(mem, MS, 0, -1, -1);
        batch.n_tokens = 1;
        batch.token[0] = p.prefix[s]; batch.pos[0] = s; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 0;
        cap.enabled = true;
        cap.rows = 0;
        if (llama_decode(ctx, batch) != 0) return fail("steer: decode failed (set)");
        cap.enabled = false;
        if (cap.rows != 1) return fail("steer: l_out capture failed (set)");
        double d = 0.0, s2 = 0.0;
        for (int e = 0; e < n_embd; ++e) { d += (double) cap.buf[e] * u[e]; s2 += (double) cap.buf[e] * cap.buf[e]; }
        out.proj0 = (float) d;
        out.norm0 = (float) std::sqrt(s2);
        set_push_val = (float) (p.set_cos * std::sqrt(s2) - d);
        llama_memory_seq_rm(mem, 0, -1, -1);
        if (std::fabs(set_push_val) > g_steer.max_dose * h_norm) return fail("steer: the set value needs a push beyond --steer-max-dose");
    }

    // common random numbers: one uniform per (sequence, step)
    std::vector<std::vector<double>> uni(NS, std::vector<double>(G));
    for (int k = 0; k < NS; ++k) {
        std::mt19937_64 rng(p.seed * 1000003ull + (uint64_t) k * 104729ull);
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        for (auto & x : uni[k]) x = ud(rng);
    }
    std::vector<std::vector<float>> lp(NS, std::vector<float>(n_vocab));

    // ---- sampled continuations, one arm at a time (the push applies to the whole batch) ----
    bool cancelled = false;
    for (int a = 0; a < n_arm && !cancelled; ++a) {
        server_steer_arm & arm = out.arms[a];
        arm.dose = doses[a];
        arm.push = doses[a] * h_norm;
        if (a == 0 && !p.baseline && !p.set && n_arm > 1) { arm.seqs.assign(NS, server_steer_seq()); continue; } // no control arm
        if (p.set && a > 0) { arm.push = set_push_val; arm.dose = set_push_val / h_norm; }
        arm.seqs.assign(NS, server_steer_seq());
        if (!set_push(ctx, u, layer, arm.push, n_embd, n_layer, cvec)) return fail("steer: cvec set failed");
        std::vector<int> act(NS), hot(NS, 0);
        for (int k = 0; k < NS; ++k) {
            const int sq = a * NS + k;
            llama_memory_seq_rm(mem, sq, -1, -1);
            if (s > 0) llama_memory_seq_cp(mem, MS, sq, -1, -1);
            act[k] = k;
            batch.token[k] = p.prefix[s]; batch.pos[k] = s; batch.n_seq_id[k] = 1; batch.seq_id[k][0] = sq; batch.logits[k] = 1;
        }
        batch.n_tokens = NS;
        for (int step = 0; step < G && !act.empty(); ++step) {
            const float push_now = (p.impulse && step > 0) ? 0.0f : arm.push;
            cap.enabled = true;
            cap.rows = 0;
            if (llama_decode(ctx, batch) != 0) return fail("steer: decode failed (generation)");
            cap.enabled = false;
            const int na = (int) act.size();
            if (cap.rows != na) return fail("steer: l_out capture failed (generation)");
            if (p.impulse && step == 0 && arm.push != 0.0f && !set_push(ctx, u, layer, 0.0f, n_embd, n_layer, cvec)) {
                return fail("steer: cvec clear failed");
            }
            std::vector<const float *> zr(na);
            for (int j = 0; j < na; ++j) zr[j] = llama_get_logits_ith(ctx, j);
            parallel_for(na, n_threads, [&](int j) {
                server_steer_seq & sq = arm.seqs[act[j]];
                const float * h = cap.buf.data() + (size_t) j * n_embd;
                double d = 0.0, s2 = 0.0;
                for (int e = 0; e < n_embd; ++e) {
                    const double x = (double) h[e] - (double) push_now * u[e]; // pre-push state
                    d += x * u[e];
                    s2 += x * x;
                }
                sq.proj.push_back((float) d);
                sq.norm.push_back((float) std::sqrt(s2));
                float * l = lp[j].data();
                log_softmax(zr[j], l, n_vocab, p.temp);
                double H = 0.0, tot = 0.0;
                for (int v = 0; v < n_vocab; ++v) {
                    const double pv = std::exp((double) l[v]);
                    H -= pv * l[v];
                    tot += pv;
                }
                const double target = uni[act[j]][step] * tot;
                double c = 0.0;
                int tok = n_vocab - 1;
                for (int v = 0; v < n_vocab; ++v) {
                    c += std::exp((double) l[v]);
                    if (c >= target) { tok = v; break; }
                }
                sq.tok.push_back(tok);
                sq.ent.push_back((float) H);
            });
            std::vector<int> nxt;
            for (int j = 0; j < na; ++j) {
                const int k = act[j];
                server_steer_seq & sq = arm.seqs[k];
                if (p.on_token && !p.on_token(a, k, sq.tok.back())) cancelled = true;
                hot[k] = sq.ent.back() > INCOH_H ? hot[k] + 1 : 0;
                if (hot[k] >= INCOH_RUN) { sq.endpoint = 2; continue; }
                if (find_loop(sq.tok, LOOP_MAX_P, LOOP_MIN_SPAN) > 0) { sq.endpoint = 1; continue; }
                if (llama_vocab_is_eog(vocab, sq.tok.back())) continue; // end of document: stop, not an endpoint
                nxt.push_back(k);
            }
            act = cancelled ? std::vector<int>() : nxt;
            for (int j = 0; j < (int) act.size(); ++j) {
                const int k = act[j];
                batch.token[j] = arm.seqs[k].tok.back(); batch.pos[j] = s + 1 + step; batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = a * NS + k; batch.logits[j] = 1;
            }
            batch.n_tokens = (int) act.size();
        }
        for (int k = 0; k < NS; ++k) llama_memory_seq_rm(mem, a * NS + k, -1, -1);
        if (a > 0 && !out.arms[0].seqs.empty() && !(out.arms[0].seqs[0].tok.empty() && !p.baseline)) {
            for (int k = 0; k < NS; ++k) {
                const auto & t0 = out.arms[0].seqs[k].tok;
                const auto & t1 = arm.seqs[k].tok;
                const int m = (int) std::min(t0.size(), t1.size());
                int fd = -1;
                for (int i = 0; i < m; ++i) if (t0[i] != t1[i]) { fd = i; break; }
                if (fd < 0 && t0.size() != t1.size()) fd = m;
                arm.seqs[k].first_div = fd;
            }
        }
    }

    // ---- teacher-forced: the existing text after the push token (or just the push token), under each arm ----
    if (n_f > 0 || !p.classes.empty()) {
        std::vector<std::vector<float>> base((size_t) n_f + 1, std::vector<float>(n_vocab));
        std::vector<std::vector<float>> cur((size_t) n_f + 1, std::vector<float>(n_vocab));
        for (int a = 0; a < n_arm; ++a) {
            server_steer_arm & arm = out.arms[a];
            auto & dst = a == 0 ? base : cur;
            llama_memory_seq_rm(mem, 0, -1, -1);
            if (s > 0) llama_memory_seq_cp(mem, MS, 0, -1, -1);
            if (!set_push(ctx, u, layer, arm.push, n_embd, n_layer, cvec)) return fail("steer: cvec set failed");
            // impulse: the push token alone under the push, then the rest unsteered
            const int n_first = p.impulse ? 1 : n_f + 1;
            for (int part = 0, k0 = 0; k0 <= n_f; ++part) {
                if (part > 0 && n_f == 0) break;
                const int n = part == 0 ? n_first : n_f + 1 - k0;
                batch.n_tokens = n;
                for (int j = 0; j < n; ++j) {
                    const int k = k0 + j;
                    batch.token[j] = k == 0 ? p.prefix[s] : p.follow[k - 1]; batch.pos[j] = s + k; batch.n_seq_id[j] = 1;
                    batch.seq_id[j][0] = 0; batch.logits[j] = 1;
                }
                if (llama_decode(ctx, batch) != 0) return fail("steer: decode failed (teacher-forced)");
                std::vector<const float *> zr(n);
                for (int j = 0; j < n; ++j) zr[j] = llama_get_logits_ith(ctx, j);
                parallel_for(n, n_threads, [&](int j) { log_softmax(zr[j], dst[k0 + j].data(), n_vocab, 1.0f); });
                k0 += n;
                if (part == 0 && p.impulse && !set_push(ctx, u, layer, 0.0f, n_embd, n_layer, cvec)) return fail("steer: cvec clear failed");
            }
            llama_memory_seq_rm(mem, 0, -1, -1);
            arm.tf.class_logp.assign(p.classes.size(), std::vector<float>(n_f + 1));
            parallel_for(n_f + 1, n_threads, [&](int k) {
                for (size_t c = 0; c < p.classes.size(); ++c) {
                    double m = -INFINITY, acc = 0.0;
                    for (int32_t t : p.classes[c].second) if (t >= 0 && t < n_vocab) m = std::max(m, (double) dst[k][t]);
                    if (!std::isfinite(m)) { arm.tf.class_logp[c][k] = -INFINITY; continue; }
                    for (int32_t t : p.classes[c].second) if (t >= 0 && t < n_vocab) acc += std::exp((double) dst[k][t] - m);
                    arm.tf.class_logp[c][k] = (float) (m + std::log(acc));
                }
            });
            if (a == 0) continue;
            arm.tf.kl.assign(n_f + 1, 0.0f);
            arm.tf.dlogp.assign(n_f, 0.0f);
            parallel_for(n_f + 1, n_threads, [&](int k) {
                double kl = 0.0;
                for (int v = 0; v < n_vocab; ++v) kl += std::exp((double) base[k][v]) * ((double) base[k][v] - cur[k][v]);
                arm.tf.kl[k] = (float) std::max(0.0, kl);
                if (k < n_f) arm.tf.dlogp[k] = cur[k][p.follow[k]] - base[k][p.follow[k]];
            });
            std::vector<std::pair<int32_t, float>> ch;
            for (int v = 0; v < n_vocab; ++v) {
                if (std::max(base[0][v], cur[0][v]) > std::log(1e-3f)) ch.emplace_back(v, cur[0][v] - base[0][v]);
            }
            std::sort(ch.begin(), ch.end(), [](const auto & x, const auto & y) { return x.second > y.second; });
            for (int i = 0; i < std::min<int>(p.n_top, (int) ch.size()); ++i) if (ch[i].second > 0) arm.tf.up.push_back(ch[i]);
            for (int i = (int) ch.size() - 1; i >= std::max(0, (int) ch.size() - p.n_top); --i) if (ch[i].second < 0) arm.tf.down.push_back(ch[i]);
        }
    }
    set_push(ctx, u, layer, 0.0f, n_embd, n_layer, cvec);
    llama_memory_clear(mem, true);
    out.sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    // ---- exposure log: one row per request with a push ----
    if (n_arm > 1) {
        long long tok = 0;
        int n_ctx_pushed = 0, n_end = 0;
        float max_dose = 0.0f;
        for (int a = 1; a < n_arm; ++a) {
            max_dose = std::max(max_dose, std::fabs(doses[a]));
            for (const auto & sq : out.arms[a].seqs) {
                tok += p.impulse ? 1 : (long long) sq.proj.size();
                n_ctx_pushed++;
                n_end += sq.endpoint != 0;
            }
            if (n_f > 0 || !p.classes.empty()) { tok += p.impulse ? 1 : n_f + 1; n_ctx_pushed++; }
        }
        std::string ds;
        for (int a = 1; a < n_arm; ++a) { char b[32]; snprintf(b, sizeof(b), "%s%+.3g", ds.empty() ? "" : " ", doses[a]); ds += b; }
        const std::string mode = p.set ? "set u.h/|h| at the push token (impulse)" : p.impulse ? "impulse (push token only)" : "step (from the push token on)";
        if (!p.log_group.empty()) {
            steer_group & g = g_steer.grp;
            const auto now = std::chrono::steady_clock::now();
            if (g.n_req > 0 && (g.kind != "steer" || g.name != p.log_group || g.mode != mode)) flush_group();
            if (g.n_req == 0) { g.kind = "steer"; g.name = p.log_group; g.mode = mode; g.model = g_steer.model_name; g.note = p.note; g.t0 = now; g.h_lo = g.h_hi = h_norm; }
            const std::string dn = p.dir.empty() ? "raw vectors" : p.dir;
            if (std::find(g.dirs.begin(), g.dirs.end(), dn) == g.dirs.end()) g.dirs.push_back(dn);
            g.tok += tok; g.n_ctx += n_ctx_pushed; g.n_end += n_end; g.n_req++;
            g.max_dose = std::max(g.max_dose, max_dose);
            g.h_lo = std::min(g.h_lo, h_norm); g.h_hi = std::max(g.h_hi, h_norm);
            if (p.log_flush || g.n_req >= 100 || now - g.t0 > std::chrono::minutes(10)) flush_group();
        } else {
            char manip[512];
            snprintf(manip, sizeof(manip), "steer after block %d along %s (unit) x dose x mean|h| %.1f; doses %s; %s", layer,
                     p.dir.empty() ? "a raw vector" : p.dir.c_str(), h_norm, ds.c_str(), mode.c_str());
            char row[2048];
            snprintf(row, sizeof(row), "%s,loom-steer,%s,%s,%.3f,%lld,%d,%s,%.3f,%s\n", today().c_str(), csv_q(g_steer.model_name).c_str(), csv_q(manip).c_str(),
                     max_dose, tok, n_ctx_pushed, csv_q(ENDPOINT_DESC).c_str(), n_ctx_pushed ? (double) n_end / n_ctx_pushed : 0.0,
                     csv_q("llama-server /steer (loom); " + std::to_string(NS) + " seq x " + std::to_string(G) + " tokens per arm" +
                           (n_f > 0 ? "; teacher-forced " + std::to_string(n_f) + " tokens" : "") + (p.note.empty() ? "" : "; " + p.note)).c_str());
            write_log_row(row);
        }
    } else if (p.log_flush) {
        flush_group();
    }
    return true;
}

bool server_guide_run(const llama_vocab * vocab, const server_guide_params & p, server_guide_result & out, std::string & err) {
    std::lock_guard<std::mutex> lock(g_steer.mutex);
    const auto t_start = std::chrono::steady_clock::now();
    if (!g_steer.model) { err = "guide: not initialized (model reloading?)"; return false; }
    const int n_embd  = llama_model_n_embd(g_steer.model);
    const int n_layer = llama_model_n_layer(g_steer.model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // the metric; probes are summed over the layers in on (lin: all; readout, dir: their layer)
    const bool is_lin   = p.kind == "lin" || p.kind == "lind" || p.kind == "lintd";
    const bool is_probe = is_lin || p.kind == "readout" || p.kind == "dir";
    const bool is_next  = p.kind == "entropy" || p.kind == "branches" || p.kind == "runner_up";
    if (!is_probe && !is_next && p.kind != "surprisal" && p.kind != "gap") { err = "guide: unknown metric kind '" + p.kind + "'"; return false; }
    std::vector<float> w;
    std::vector<char> on;
    float bias = 0.0f;
    if (is_lin) {
        if (!server_jlens_lin_get(p.name, w, bias)) { err = "guide: unknown all-layer probe '" + p.name + "' (start the server with --lin-probes)"; return false; }
        if (w.size() != (size_t) n_layer * n_embd) { err = "guide: the all-layer probes do not match the model"; return false; }
        on.assign(n_layer, 1);
    } else if (is_probe) {
        int L = -1;
        float h_norm = 0.0f;
        std::vector<float> u;
        const bool ok = p.kind == "readout" ? server_jlens_readout_get(p.name, L, u, bias) : server_jlens_dir_get(p.name, L, h_norm, u);
        if (!ok) { err = "guide: unknown " + std::string(p.kind == "readout" ? "readout '" : "direction '") + p.name + "'"; return false; }
        if (L < 0 || L >= n_layer || (int) u.size() != n_embd) { err = "guide: the probe does not match the model"; return false; }
        w.assign((size_t) n_layer * n_embd, 0.0f);
        std::copy(u.begin(), u.end(), w.begin() + (size_t) L * n_embd);
        on.assign(n_layer, 0);
        on[L] = 1;
    }
    if (p.pct_q.size() < 2 || p.pct_q.size() != p.pct_v.size() || !std::is_sorted(p.pct_q.begin(), p.pct_q.end())) {
        err = "guide: the color scale needs two or more (percentile, value) pairs in ascending order";
        return false;
    }
    const int s = (int) p.prefix.size() - 1;
    if (s < 0) { err = "guide: empty prefix"; return false; }
    if (p.k < 1 || p.k > N_ROWS_1U - 1) { err = "guide: k must be in [1, " + std::to_string(N_ROWS_1U - 1) + "]"; return false; }
    if (!(p.top_p > 0.0f && p.top_p <= 1.0f)) { err = "guide: top_p must be in (0, 1]"; return false; }
    if (!(p.temp > 0.0f)) { err = "guide: temperature must be > 0"; return false; }
    if (!std::isfinite(p.beta) || std::fabs(p.beta) > 50.0f) { err = "guide: |beta| must be <= 50"; return false; }
    if (p.n_gen < 1 || s + 1 + p.n_gen + p.k > N_CTX) {
        err = "guide: need n_gen >= 1 and text + n_gen + k <= " + std::to_string(N_CTX) + " (the steering context)";
        return false;
    }

    if (!ensure_ctx(err)) return false;
    llama_context * ctx = g_steer.ctx;
    llama_memory_t  mem = llama_get_memory(ctx);
    steer_capture & cap = g_steer.cap;
    cap.n_embd  = n_embd;
    cap.enabled = false;
    cap.lin_w   = w.data();
    cap.lin_on  = on;
    llama_memory_clear(mem, true);
    if (llama_set_adapter_cvec(ctx, nullptr, 0, n_embd, 0, 0) != 0) { err = "guide: cvec clear failed"; return false; }
    llama_batch batch = llama_batch_init(std::max(N_BATCH, N_SEQ_MAX), 0, 1);
    struct run_guard {
        llama_batch & b; steer_capture & c; llama_memory_t m;
        ~run_guard() { llama_batch_free(b); c.enabled = false; c.lin_on.clear(); c.lin_w = nullptr; llama_memory_clear(m, true); }
    } rg{batch, cap, mem};
    // decode the batch; with a probe, capture its per-row sums (rows must match)
    auto decode = [&](int n_rows, const char * what) {
        cap.rows = 0;
        cap.lin_acc.clear();
        cap.lin_h2.clear();
        cap.enabled = is_probe;
        const bool ok = llama_decode(ctx, batch) == 0;
        cap.enabled = false;
        if (!ok) { err = std::string("guide: decode failed (") + what + ")"; return false; }
        if (is_probe && cap.rows != n_rows) { err = std::string("guide: l_out capture failed (") + what + ")"; return false; }
        return true;
    };

    // the text up to the last token, then the last token alone: its logits and metric value (the previous value of a change)
    for (int cur = 0; cur < s; cur += N_UBATCH) {
        const int n = std::min(N_UBATCH, s - cur);
        batch.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            batch.token[j] = p.prefix[cur + j]; batch.pos[j] = cur + j; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = 0;
        }
        cap.enabled = false;
        if (llama_decode(ctx, batch) != 0) { err = "guide: decode failed (prefix)"; return false; }
    }
    batch.n_tokens = 1;
    batch.token[0] = p.prefix[s]; batch.pos[0] = s; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
    if (!decode(1, "last token")) return false;
    double f_prev = is_probe ? cap.lin_acc[0] + bias : 0.0;
    std::vector<float> zcur(llama_get_logits_ith(ctx, 0), llama_get_logits_ith(ctx, 0) + n_vocab);

    const int n_threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    std::mt19937_64 rng(p.seed);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::vector<float> l1(n_vocab), lT(n_vocab);
    std::vector<int32_t> order(n_vocab), cand;
    int hot = 0;
    for (int step = 0; step < p.n_gen; ++step) {
        const int pos = s + 1 + step;
        log_softmax(zcur.data(), l1.data(), n_vocab, 1.0f);
        log_softmax(zcur.data(), lT.data(), n_vocab, p.temp);
        double H1 = 0.0, HT = 0.0;
        for (int v = 0; v < n_vocab; ++v) { H1 -= std::exp((double) l1[v]) * l1[v]; HT -= std::exp((double) lT[v]) * lT[v]; }

        // candidates: most likely first, until top_p of the probability or k of them
        std::iota(order.begin(), order.end(), 0);
        const int kk = std::min(p.k, n_vocab);
        std::partial_sort(order.begin(), order.begin() + kk, order.end(), [&](int32_t x, int32_t y) { return lT[x] > lT[y]; });
        cand.clear();
        double cum = 0.0;
        for (int i = 0; i < kk; ++i) {
            cand.push_back(order[i]);
            cum += std::exp((double) lT[order[i]]);
            if (cum >= p.top_p) break;
        }
        const int nc = (int) cand.size();

        // every candidate in its own sequence on the text so far, as one batch
        for (int j = 0; j < nc; ++j) {
            llama_memory_seq_rm(mem, 1 + j, -1, -1);
            llama_memory_seq_cp(mem, 0, 1 + j, -1, -1);
            batch.token[j] = cand[j]; batch.pos[j] = pos; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 1 + j; batch.logits[j] = 1;
        }
        batch.n_tokens = nc;
        if (!decode(nc, "candidates")) return false;
        std::vector<const float *> zr(nc);
        for (int j = 0; j < nc; ++j) zr[j] = llama_get_logits_ith(ctx, j);

        std::vector<double> val(nc), z(nc);
        if (is_next) {
            // next-token statistics after each candidate, as server_jlens_compute's token stats
            parallel_for(nc, n_threads, [&](int j) {
                const float * zz = zr[j];
                double mx = -INFINITY, mx2 = -INFINITY;
                for (int x = 0; x < n_vocab; ++x) {
                    if (zz[x] > mx) { mx2 = mx; mx = zz[x]; } else if (zz[x] > mx2) { mx2 = zz[x]; }
                }
                double se = 0.0, sze = 0.0, se2 = 0.0;
                for (int x = 0; x < n_vocab; ++x) {
                    const double e = std::exp((double) zz[x] - mx);
                    se += e; sze += e * ((double) zz[x] - mx); se2 += e * e;
                }
                val[j] = p.kind == "entropy" ? std::log(se) - sze / se : p.kind == "branches" ? se * se / se2 : std::exp(mx2 - mx);
            });
        }
        for (int j = 0; j < nc; ++j) {
            const double f = is_probe ? cap.lin_acc[j] + bias : 0.0;
            const double c = -(double) l1[cand[j]];
            if (p.kind == "lin" || p.kind == "readout") val[j] = f;
            else if (p.kind == "lind")      val[j] = f - f_prev;
            else if (p.kind == "lintd")     val[j] = -c - p.horizon * (f - f_prev);
            else if (p.kind == "dir")       val[j] = cap.lin_h2[j] > 0.0 ? cap.lin_acc[j] / std::sqrt(cap.lin_h2[j]) : 0.0;
            else if (p.kind == "surprisal") val[j] = c;
            else if (p.kind == "gap")       val[j] = c - H1;
            z[j] = probit(scale_pct(p.pct_q, p.pct_v, val[j]) / 100.0);
        }

        // choose: the best candidate, or sample from p(x) exp(beta sign z) over the candidates
        double pm = 0.0, zm = 0.0;
        for (int j = 0; j < nc; ++j) { pm += std::exp((double) lT[cand[j]]); zm += std::exp((double) lT[cand[j]]) * z[j]; }
        zm /= pm;
        int ch = 0;
        if (p.maximize) {
            for (int j = 1; j < nc; ++j) if (p.sign * val[j] > p.sign * val[ch]) ch = j;
        } else {
            std::vector<double> lw(nc);
            for (int j = 0; j < nc; ++j) lw[j] = (double) lT[cand[j]] + (double) p.beta * p.sign * z[j];
            const double m = *std::max_element(lw.begin(), lw.end());
            double tot = 0.0;
            for (auto & x : lw) { x = std::exp(x - m); tot += x; }
            const double target = ud(rng) * tot;
            double acc = 0.0;
            ch = nc - 1;
            for (int j = 0; j < nc; ++j) { acc += lw[j]; if (acc >= target) { ch = j; break; } }
        }

        // keep the chosen candidate in sequence 0; drop the others
        llama_memory_seq_cp(mem, 1 + ch, 0, pos, pos + 1);
        for (int j = 0; j < nc; ++j) llama_memory_seq_rm(mem, 1 + j, -1, -1);
        zcur.assign(zr[ch], zr[ch] + n_vocab);
        if (is_probe) f_prev = cap.lin_acc[ch] + bias;

        const int32_t tok = cand[ch];
        out.tok.push_back(tok);
        out.value.push_back((float) val[ch]);
        out.z.push_back((float) z[ch]);
        out.z_model.push_back((float) zm);
        out.ent.push_back((float) HT);
        out.n_cand.push_back(nc);
        if (p.on_token && !p.on_token(tok, (float) val[ch], (float) z[ch])) break;
        hot = HT > INCOH_H ? hot + 1 : 0;
        if (hot >= INCOH_RUN) { out.endpoint = 2; break; }
        if (find_loop(out.tok, LOOP_MAX_P, LOOP_MIN_SPAN) > 0) { out.endpoint = 1; break; }
        if (llama_vocab_is_eog(vocab, tok)) break; // end of document: stop, not an endpoint
    }
    out.sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    // ---- exposure log ----
    const std::string metric = p.kind + (p.name.empty() ? "" : ":" + p.name) + (p.sign < 0 ? " (flipped)" : "");
    const std::string mode = p.maximize ? "maximize" : "tilt";
    if (!p.log_group.empty()) {
        steer_group & g = g_steer.grp;
        const auto now = std::chrono::steady_clock::now();
        if (g.n_req > 0 && (g.kind != "guide" || g.name != p.log_group || g.mode != mode)) flush_group();
        if (g.n_req == 0) { g.kind = "guide"; g.name = p.log_group; g.mode = mode; g.model = g_steer.model_name; g.note = p.note; g.t0 = now; }
        if (std::find(g.dirs.begin(), g.dirs.end(), metric) == g.dirs.end()) g.dirs.push_back(metric);
        g.tok += (long long) out.tok.size(); g.n_ctx++; g.n_end += out.endpoint != 0; g.n_req++;
        if (!p.maximize) g.max_beta = std::max(g.max_beta, std::fabs(p.beta));
        if (p.log_flush || g.n_req >= 100 || now - g.t0 > std::chrono::minutes(10)) flush_group();
    } else {
        char manip[512];
        snprintf(manip, sizeof(manip), "probe-guided sampling, no push: each token chosen among the top candidates (top_p %.2f, k <= %d) by %s; %s",
                 p.top_p, p.k, metric.c_str(), p.maximize ? "maximize" : ("tilt p(x) exp(beta z), beta " + std::to_string(p.beta).substr(0, 5)).c_str());
        char row[2048];
        snprintf(row, sizeof(row), "%s,loom-guide,%s,%s,%.3f,%lld,%d,%s,%.3f,%s\n", today().c_str(), csv_q(g_steer.model_name).c_str(), csv_q(manip).c_str(),
                 0.0, (long long) out.tok.size(), 1, csv_q(ENDPOINT_DESC).c_str(), out.endpoint != 0 ? 1.0 : 0.0,
                 csv_q("llama-server /steer (guide)" + (p.note.empty() ? "" : "; " + p.note)).c_str());
        write_log_row(row);
    }
    return true;
}

// one row of the input embedding matrix, dequantized (the matrix may live on the GPU)
static bool embd_row(const ggml_tensor * t, int32_t tok, float * out, std::vector<uint8_t> & raw) {
    const int64_t n  = t->ne[0];
    const size_t  rs = ggml_row_size(t->type, n);
    raw.resize(rs);
    ggml_backend_tensor_get(t, raw.data(), (size_t) tok * t->nb[1], rs);
    if (t->type == GGML_TYPE_F32) { std::memcpy(out, raw.data(), (size_t) n * sizeof(float)); return true; }
    const auto * tr = ggml_get_type_traits(t->type);
    if (!tr->to_float) return false;
    tr->to_float(raw.data(), out, n);
    return true;
}

bool server_superpose_run(const llama_vocab * vocab, const server_superpose_params & p, server_superpose_result & out, std::string & err) {
    std::lock_guard<std::mutex> lock(g_steer.mutex);
    const auto t_start = std::chrono::steady_clock::now();
    if (!g_steer.model) { err = "superpose: not initialized (model reloading?)"; return false; }
    const int n_embd  = llama_model_n_embd(g_steer.model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const ggml_tensor * E = g_steer.model->tok_embd;
    if (!E || E->ne[0] != n_embd || E->ne[1] < n_vocab || !E->buffer) { err = "superpose: the model has no usable input embedding matrix"; return false; }
    const int s = (int) p.prefix.size() - 1;
    if (s < 0) { err = "superpose: empty prefix"; return false; }
    if (p.k < 1 || p.k > SUP_K_MAX) { err = "superpose: k must be in [1, " + std::to_string(SUP_K_MAX) + "]"; return false; }
    if (!(p.top_p > 0.0f && p.top_p <= 1.0f)) { err = "superpose: top_p must be in (0, 1]"; return false; }
    if (!(p.temp > 0.0f)) { err = "superpose: temperature must be > 0"; return false; }
    if (p.n_show < 1 || p.n_show > 64) { err = "superpose: n_show must be in [1, 64]"; return false; }
    const bool resid = p.input == "resid" || p.input == "resid_norm";
    if (!resid && p.input != "mix") { err = "superpose: input must be mix, resid or resid_norm"; return false; }
    if (p.mix && (p.mix_k < 1 || p.mix_k > N_ROWS_1U)) { err = "superpose: mix_k must be in [1, " + std::to_string(N_ROWS_1U) + "]"; return false; }
    if (p.n_gen < 1 || s + 1 + p.n_gen + p.mix_k > N_CTX) {
        err = "superpose: need n_gen >= 1 and text + n_gen + mix_k <= " + std::to_string(N_CTX) + " (the steering context)";
        return false;
    }

    if (!ensure_ctx(err)) return false;
    llama_context * ctx = g_steer.ctx;
    llama_memory_t  mem = llama_get_memory(ctx);
    steer_capture & cap = g_steer.cap;
    cap.enabled = false;
    cap.lin_on.clear();
    cap.n_embd  = n_embd;
    cap.name    = p.input == "resid" ? "l_out-" + std::to_string(llama_model_n_layer(g_steer.model) - 1) : "result_norm";
    llama_memory_clear(mem, true);
    if (llama_set_adapter_cvec(ctx, nullptr, 0, n_embd, 0, 0) != 0) { err = "superpose: cvec clear failed"; return false; }
    llama_batch tb = llama_batch_init(N_BATCH, 0, 1);
    llama_batch eb = llama_batch_init(N_ROWS_1U + 1, n_embd, 1); // raw embeddings: the superposed input + each candidate
    struct run_guard {
        llama_batch & t; llama_batch & e; llama_memory_t m; steer_capture & c;
        ~run_guard() { llama_batch_free(t); llama_batch_free(e); llama_memory_clear(m, true); c.enabled = false; }
    } rg{tb, eb, mem, cap};
    // resid: decode with the last block's output (or the final norm's) captured; fb = its row for the superposed input
    std::vector<float> fb(n_embd);
    auto decode_cap = [&](llama_batch & b, int row, const char * what) {
        cap.rows = 0;
        cap.enabled = resid;
        const bool ok = llama_decode(ctx, b) == 0;
        cap.enabled = false;
        if (!ok) { err = std::string("superpose: decode failed (") + what + ")"; return false; }
        if (resid) {
            if (cap.rows <= row) { err = std::string("superpose: residual capture failed (") + what + ")"; return false; }
            std::copy(cap.buf.begin() + (size_t) row * n_embd, cap.buf.begin() + (size_t) (row + 1) * n_embd, fb.begin());
        }
        return true;
    };

    for (int cur = 0; cur <= s; cur += N_UBATCH) {
        const int n = std::min(N_UBATCH, s + 1 - cur);
        tb.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            tb.token[j] = p.prefix[cur + j]; tb.pos[j] = cur + j; tb.n_seq_id[j] = 1; tb.seq_id[j][0] = 0; tb.logits[j] = cur + j == s;
        }
        if (cur + n > s) { if (!decode_cap(tb, 0, "prefix")) return false; } // the only output row: the last token
        else if (llama_decode(ctx, tb) != 0) { err = "superpose: decode failed (prefix)"; return false; }
    }
    std::vector<float> zcur(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + n_vocab);

    const int n_threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    const int n_chunk   = 64; // vocabulary chunks for the parallel sums
    std::vector<float> lT(n_vocab), cand_e((size_t) p.k * n_embd), in(n_embd);
    std::vector<float> P((size_t) (1 + (p.mix ? std::min(p.k, p.mix_k) : 0)) * n_vocab); // row 0: q, row 1 + j: p_x of candidate j
    std::vector<uint8_t> raw;
    std::vector<int32_t> order(n_vocab), cand, argmax_toks;
    std::vector<double> w;
    int hot = 0;
    for (int step = 0; step < p.n_gen; ++step) {
        const int pos = s + 1 + step;
        log_softmax(zcur.data(), lT.data(), n_vocab, p.temp);
        double HT = 0.0;
        for (int v = 0; v < n_vocab; ++v) HT -= std::exp((double) lT[v]) * lT[v];

        // candidates: most likely first, until top_p of the probability or k of them; weights renormalized over them
        std::iota(order.begin(), order.end(), 0);
        const int kk = std::min(p.k, n_vocab);
        std::partial_sort(order.begin(), order.begin() + kk, order.end(), [&](int32_t x, int32_t y) { return lT[x] > lT[y]; });
        cand.clear();
        double mass = 0.0;
        for (int i = 0; i < kk; ++i) {
            cand.push_back(order[i]);
            mass += std::exp((double) lT[order[i]]);
            if (mass >= p.top_p) break;
        }
        const int nc = (int) cand.size();
        w.assign(nc, 0.0);
        double sw2 = 0.0;
        for (int j = 0; j < nc; ++j) { w[j] = std::exp((double) lT[cand[j]]) / mass; sw2 += w[j] * w[j]; }

        // the superposed input: sum w e(x), optionally rescaled to the weighted mean length of the e(x); or the residual
        std::fill(in.begin(), in.end(), 0.0f);
        double len_w = 0.0;
        for (int j = 0; j < nc; ++j) {
            float * e = cand_e.data() + (size_t) j * n_embd;
            if (!embd_row(E, cand[j], e, raw)) { err = "superpose: cannot dequantize the input embeddings"; return false; }
            double e2 = 0.0;
            for (int d = 0; d < n_embd; ++d) { in[d] += (float) w[j] * e[d]; e2 += (double) e[d] * e[d]; }
            len_w += w[j] * std::sqrt(e2);
        }
        auto dot = [&](const float * a, const float * b) { double r = 0.0; for (int d = 0; d < n_embd; ++d) r += (double) a[d] * b[d]; return r; };
        const std::vector<float> mixv = in;
        if (resid) {
            in = fb;
            if (p.input == "resid_norm") {
                const double f2 = dot(in.data(), in.data());
                if (f2 > 0.0) { const float sc = (float) (len_w / std::sqrt(f2)); for (auto & x : in) x *= sc; }
            }
        }
        double in2 = dot(in.data(), in.data());
        const double shrink = len_w > 0.0 ? std::sqrt(in2) / len_w : 1.0;
        if (!resid && p.norm && in2 > 0.0) {
            const float sc = (float) (len_w / std::sqrt(in2));
            for (auto & x : in) x *= sc;
            in2 = dot(in.data(), in.data());
        }
        const double t2 = dot(cand_e.data(), cand_e.data()), m2 = dot(mixv.data(), mixv.data());

        // one batch: the superposed input continues sequence 0; with mix, each of the top mix_k candidates is decoded
        // alone in its own sequence on the same past
        server_superpose_step st;
        st.tok    = cand[0];
        st.ent    = (float) HT;
        st.eff    = (float) (1.0 / sw2);
        st.mass   = (float) mass;
        st.n_cand = nc;
        st.shrink = (float) shrink;
        if (in2 > 0.0 && t2 > 0.0) st.cos_top = (float) (dot(in.data(), cand_e.data()) / std::sqrt(in2 * t2));
        if (in2 > 0.0 && m2 > 0.0) st.cos_mix = (float) (dot(in.data(), mixv.data()) / std::sqrt(in2 * m2));
        for (int j = 0; j < std::min(nc, p.n_show); ++j) { st.top.push_back(cand[j]); st.top_w.push_back((float) w[j]); }
        const int M = p.mix ? std::min(nc, p.mix_k) : 0;
        for (int r = 0; r <= M; ++r) {
            std::memcpy(eb.embd + (size_t) r * n_embd, r == 0 ? in.data() : cand_e.data() + (size_t) (r - 1) * n_embd, sizeof(float) * n_embd);
            if (r > 0) {
                llama_memory_seq_rm(mem, r, -1, -1);
                llama_memory_seq_cp(mem, 0, r, -1, -1);
            }
            eb.pos[r] = pos; eb.n_seq_id[r] = 1; eb.seq_id[r][0] = r; eb.logits[r] = 1;
        }
        eb.n_tokens = 1 + M;
        const bool ok = decode_cap(eb, 0, "step");
        for (int r = 1; r <= M; ++r) llama_memory_seq_rm(mem, r, -1, -1);
        if (!ok) return false;
        zcur.assign(llama_get_logits_ith(ctx, 0), llama_get_logits_ith(ctx, 0) + n_vocab);
        if (M > 0) {
            // next-token distributions (temperature 1): q after the superposed input, p_x after each checked candidate;
            // the mixture is renormalized over the checked candidates (mix_mass of the probability)
            parallel_for(1 + M, n_threads, [&](int r) {
                float * pr = P.data() + (size_t) r * n_vocab;
                log_softmax(llama_get_logits_ith(ctx, r), pr, n_vocab, 1.0f);
                for (int v = 0; v < n_vocab; ++v) pr[v] = std::exp(pr[v]);
            });
            const float * q = P.data();
            double wm = 0.0;
            for (int j = 0; j < M; ++j) wm += w[j];
            std::vector<double> kl_c(n_chunk, 0.0), tv_c(n_chunk, 0.0);
            parallel_for(n_chunk, n_threads, [&](int c) {
                const int v0 = (int) ((int64_t) n_vocab * c / n_chunk), v1 = (int) ((int64_t) n_vocab * (c + 1) / n_chunk);
                double kl = 0.0, tv = 0.0;
                for (int v = v0; v < v1; ++v) {
                    double m = 0.0;
                    for (int j = 0; j < M; ++j) m += w[j] * P[(size_t) (1 + j) * n_vocab + v];
                    m /= wm;
                    if (m > 0.0) kl += m * (std::log(m) - std::log(std::max((double) q[v], 1e-30)));
                    tv += std::fabs(m - q[v]);
                }
                kl_c[c] = kl; tv_c[c] = tv;
            });
            std::vector<double> tv_j(M, 0.0);
            parallel_for(M, n_threads, [&](int j) {
                const float * pj = P.data() + (size_t) (1 + j) * n_vocab;
                double tv = 0.0;
                for (int v = 0; v < n_vocab; ++v) tv += std::fabs((double) pj[v] - q[v]);
                tv_j[j] = 0.5 * tv;
            });
            const int near = (int) (std::min_element(tv_j.begin(), tv_j.end()) - tv_j.begin());
            st.kl       = (float) std::max(0.0, std::accumulate(kl_c.begin(), kl_c.end(), 0.0));
            st.tv       = (float) (0.5 * std::accumulate(tv_c.begin(), tv_c.end(), 0.0));
            st.tv_top   = (float) tv_j[0];
            st.near     = cand[near];
            st.near_w   = (float) w[near];
            st.tv_near  = (float) tv_j[near];
            st.mix_mass = (float) (wm * mass);
        }

        out.steps.push_back(st);
        argmax_toks.push_back(st.tok);
        if (p.on_step && !p.on_step(st)) break;
        hot = HT > INCOH_H ? hot + 1 : 0;
        if (hot >= INCOH_RUN) { out.endpoint = 2; break; }
        if (find_loop(argmax_toks, LOOP_MAX_P, LOOP_MIN_SPAN) > 0) { out.endpoint = 1; break; }
        if (llama_vocab_is_eog(vocab, st.tok)) break; // end of document: stop, not an endpoint
    }
    out.sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    // ---- exposure log ----
    char mode[320];
    snprintf(mode, sizeof(mode), "%s; top_p %.2f, k <= %d, temp %.2f%s%s", p.input == "mix" ? "input sum w e(x)" : p.input == "resid" ?
             "input = the last block's output at the previous position (raw)" : "input = the final-norm output at the previous position, rescaled to sum w |e(x)|",
             p.top_p, p.k, p.temp, p.norm ? ", length-matched" : "",
             p.mix ? (", the top " + std::to_string(p.mix_k) + " also decoded alone").c_str() : "");
    if (!p.log_group.empty()) {
        steer_group & g = g_steer.grp;
        const auto now = std::chrono::steady_clock::now();
        if (g.n_req > 0 && (g.kind != "superpose" || g.name != p.log_group || g.mode != mode)) flush_group();
        if (g.n_req == 0) { g.kind = "superpose"; g.name = p.log_group; g.mode = mode; g.model = g_steer.model_name; g.note = p.note; g.t0 = now; }
        g.tok += (long long) out.steps.size(); g.n_ctx++; g.n_end += out.endpoint != 0; g.n_req++;
        if (p.log_flush || g.n_req >= 100 || now - g.t0 > std::chrono::minutes(10)) flush_group();
    } else {
        char row[2048];
        snprintf(row, sizeof(row), "%s,loom-superpose,%s,%s,%.3f,%lld,%d,%s,%.3f,%s\n", today().c_str(), csv_q(g_steer.model_name).c_str(),
                 csv_q(std::string("superposed input, no push: the probability-weighted mean of the top candidates' input embeddings fed "
                                   "instead of a token at every step (off the token manifold); ") + mode).c_str(),
                 0.0, (long long) out.steps.size(), 1, csv_q(ENDPOINT_DESC).c_str(), out.endpoint != 0 ? 1.0 : 0.0,
                 csv_q("llama-server /steer (superpose)" + (p.note.empty() ? "" : "; " + p.note)).c_str());
        write_log_row(row);
    }
    return true;
}

static std::atomic<bool> g_explore_stop{false};

void server_explore_stop() { g_explore_stop = true; }

int server_explore_n_ctx() { return N_CTX; }

bool server_explore_run(const llama_vocab * vocab, const server_explore_params & p, server_explore_result & out, std::string & err) {
    std::lock_guard<std::mutex> lock(g_steer.mutex);
    g_explore_stop = false;
    const auto t_start = std::chrono::steady_clock::now();
    if (!g_steer.model) { err = "explore: not initialized (model reloading?)"; return false; }
    const int n_embd  = llama_model_n_embd(g_steer.model);
    const int n_layer = llama_model_n_layer(g_steer.model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int s = (int) p.prefix.size() - 1;
    const int W = p.walkers;
    if (s < 0) { err = "explore: empty prefix"; return false; }
    if (W < 1 || W > N_SEQ_MAX - 1) { err = "explore: walkers must be in [1, " + std::to_string(N_SEQ_MAX - 1) + "]"; return false; }
    if (p.tau < 1) { err = "explore: tau must be >= 1"; return false; }
    if (p.layer < 0 || p.layer >= n_layer) { err = "explore: layer out of range"; return false; }
    if (!(p.ema > 0.0f && p.ema <= 1.0f)) { err = "explore: ema must be in (0, 1]"; return false; }
    if (!(p.radius > 0.0f)) { err = "explore: radius must be > 0"; return false; }
    if (p.max_bins < 1 || p.max_bins > 1024) { err = "explore: max_bins must be in [1, 1024]"; return false; }
    if (!(p.alpha >= 0.1f && p.alpha <= 2.0f)) { err = "explore: alpha must be in [0.1, 2]"; return false; }
    if (p.n_ex < 0 || p.n_ex > 8) { err = "explore: n_ex must be in [0, 8]"; return false; }
    if (!(p.duty >= 0.1f && p.duty <= 1.0f)) { err = "explore: duty must be in [0.1, 1]"; return false; }
    const bool beam = p.method == "beam";
    const bool fc = p.method == "fc" || beam, plain = p.method == "plain" || (!p.resample && !fc);
    if (!fc && p.method != "we" && p.method != "plain") { err = "explore: method must be we, fc, beam or plain"; return false; }
    if (fc && (p.fc_top < 1 || p.fc_top > 1024)) { err = "explore: fc_top must be in [1, 1024]"; return false; }
    // the walkers share their histories in the KV cache (a copy shares its parent's cells, a dropped walker frees the ones
    // only it held), so what has to fit is the live tree, checked every step; here only the first step
    if (p.n_gen < 1 || s + 1 + W + 16 > N_CTX) {
        err = "explore: need n_gen >= 1 and text + walkers <= " + std::to_string(N_CTX - 16) + " (the steering context)";
        return false;
    }

    if (!ensure_ctx(err)) return false;
    llama_context * ctx = g_steer.ctx;
    llama_memory_t  mem = llama_get_memory(ctx);
    steer_capture & cap = g_steer.cap;
    cap.enabled = false;
    cap.lin_on.clear();
    cap.n_embd  = n_embd;
    cap.name    = "l_out-" + std::to_string(p.layer);
    cap.append  = true;
    llama_memory_clear(mem, true);
    if (llama_set_adapter_cvec(ctx, nullptr, 0, n_embd, 0, 0) != 0) { cap.append = false; err = "explore: cvec clear failed"; return false; }
    llama_batch batch = llama_batch_init(std::max(N_BATCH, N_SEQ_MAX), 0, 1);
    struct run_guard {
        llama_batch & b; llama_memory_t m; steer_capture & c;
        ~run_guard() { llama_batch_free(b); llama_memory_clear(m, true); c.enabled = false; c.append = false; }
    } rg{batch, mem, cap};
    auto decode_cap = [&](int n_rows, const char * what) {
        cap.rows = 0;
        cap.enabled = true;
        const bool ok = llama_decode(ctx, batch) == 0;
        cap.enabled = false;
        if (!ok) { err = std::string("explore: decode failed (") + what + ")"; return false; }
        if (cap.rows != n_rows) { err = std::string("explore: l_out capture failed (") + what + ")"; return false; }
        return true;
    };

    // the text up to the last token, then the last token alone: its next-token distribution and features
    for (int cur = 0; cur < s; cur += N_UBATCH) {
        const int n = std::min(N_UBATCH, s - cur);
        batch.n_tokens = n;
        for (int j = 0; j < n; ++j) {
            batch.token[j] = p.prefix[cur + j]; batch.pos[j] = cur + j; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = 0;
        }
        if (llama_decode(ctx, batch) != 0) { err = "explore: decode failed (prefix)"; return false; }
    }
    batch.n_tokens = 1;
    batch.token[0] = p.prefix[s]; batch.pos[0] = s; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
    if (!decode_cap(1, "last token")) return false;
    const std::vector<float> h0(cap.buf.begin(), cap.buf.begin() + n_embd);
    std::vector<float> lp0(n_vocab);
    log_softmax(llama_get_logits_ith(ctx, 0), lp0.data(), n_vocab, 1.0f);

    const int n_threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    std::mt19937_64 rng(p.seed);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    struct walker {
        int                  seq = -1;
        std::vector<int32_t> toks;
        std::vector<float>   path;     // exact log p of each token
        double               w = 0.0;
        std::vector<float>   lp, feat; // next-token log probabilities; feature moving average
        int                  bin = -1, forced = -1, hot = 0, fam = -1, node = -1;
        bool                 pushed = false; // the forced token's log p is already in path (fc/beam children)
    };
    std::vector<walker> live;
    std::vector<server_explore_walker> fin;
    std::vector<std::vector<float>>    fin_feat;
    // every walker's history as a tree: node = one token at one step (one KV cell), npar its parent (-1: the prompt)
    std::vector<int> npar, mark;
    int mark_id = 0;
    auto live_cells = [&]() {
        mark.resize(npar.size(), 0);
        ++mark_id;
        long long n = s + 1;
        for (const auto & wk : live) for (int i = wk.node; i >= 0 && mark[i] != mark_id; i = npar[i]) { mark[i] = mark_id; n++; }
        return n;
    };
    std::vector<int> free_ids;
    for (int i = N_SEQ_MAX - 1; i >= 1; --i) free_ids.push_back(i);
    auto alloc   = [&]() { const int id = free_ids.back(); free_ids.pop_back(); return id; };
    auto release = [&](int id) { llama_memory_seq_rm(mem, id, -1, -1); free_ids.push_back(id); };

    // c tokens by systematic sampling from exp(lp), in random order: each one is marginally ~ p, together they are spread
    auto systematic = [&](const std::vector<float> & lp, int c) {
        std::vector<int32_t> r;
        const double u = ud(rng);
        double cum = 0.0;
        int k = 0, best = 0;
        for (int v = 0; v < n_vocab && k < c; ++v) {
            if (lp[v] > lp[best]) best = v;
            cum += std::exp((double) lp[v]);
            while (k < c && (k + u) / c < cum) { r.push_back(v); k++; }
        }
        while (k++ < c) r.push_back(best); // rounding: the sum fell short of 1
        std::shuffle(r.begin(), r.end(), rng);
        return r;
    };

    for (int i = 0; i < (fc ? 1 : W); ++i) { // fc: one walker with all the weight; the first selection fans it out
        walker wk;
        wk.seq = alloc();
        llama_memory_seq_cp(mem, 0, wk.seq, -1, -1);
        wk.w = fc ? 1.0 : 1.0 / W; wk.lp = lp0; wk.feat = h0;
        live.push_back(std::move(wk));
    }
    llama_memory_seq_rm(mem, 0, -1, -1);
    if (!fc && !plain) {
        const auto t0 = systematic(lp0, W);
        for (int i = 0; i < W; ++i) live[i].forced = t0[i];
    }

    // ---- bins: Voronoi cells around centers made on the fly, in features standardized by running statistics ----
    std::vector<double> f_mean(n_embd, 0.0), f_m2(n_embd, 0.0);
    long long f_n = 0;
    std::vector<float> inv_sd(n_embd, 1.0f);
    std::vector<std::vector<float>> centers;
    std::vector<int> born;
    double R = -1.0;
    auto dist = [&](const std::vector<float> & a, const std::vector<float> & b) {
        double d2 = 0.0;
        for (int d = 0; d < n_embd; ++d) { const double x = (a[d] - b[d]) * inv_sd[d]; d2 += x * x; }
        return std::sqrt(d2 / n_embd);
    };
    int n_rounds = 0, last_npar = 0;
    auto do_round = [&](int step, bool resample_now, server_explore_round & rd) {
        if (beam) {
            rd = server_explore_round();
            rd.step = step; rd.n_live = (int) live.size(); rd.n_done = (int) fin.size(); rd.n_par = last_npar;
            double sw = 0.0, sw2 = 0.0;
            std::vector<std::pair<double, const std::vector<int32_t> *>> m;
            for (const auto & wk : live) { sw += wk.w; sw2 += wk.w * wk.w; m.push_back({wk.w, &wk.toks}); }
            rd.eff = sw2 > 0.0 ? sw * sw / sw2 : 0.0;
            rd.mass = sw;
            for (const auto & f : fin) { rd.mass += f.w; m.push_back({f.w, &f.toks}); }
            const int nt = std::min<int>(p.n_ex, (int) m.size());
            std::partial_sort(m.begin(), m.begin() + nt, m.end(), [](const auto & x, const auto & y) { return x.first > y.first; });
            for (int k = 0; k < nt; ++k) { rd.top.push_back(*m[k].second); rd.top_w.push_back(m[k].first); }
            n_rounds++;
            return;
        }
        for (const auto & wk : live) {
            f_n++;
            for (int d = 0; d < n_embd; ++d) { const double x = wk.feat[d], dl = x - f_mean[d]; f_mean[d] += dl / f_n; f_m2[d] += dl * (x - f_mean[d]); }
        }
        if (f_n > 1) {
            double vbar = 0.0;
            for (int d = 0; d < n_embd; ++d) vbar += f_m2[d] / (f_n - 1);
            vbar /= n_embd;
            for (int d = 0; d < n_embd; ++d) inv_sd[d] = (float) (1.0 / std::sqrt(f_m2[d] / (f_n - 1) + 1e-3 * vbar + 1e-12));
        }
        // items: the live walkers, then finished walkers without a bin yet
        std::vector<const std::vector<float> *> items;
        std::vector<int> item_of; // >= 0: live index; < 0: -1 - finished index
        for (size_t i = 0; i < live.size(); ++i) { items.push_back(&live[i].feat); item_of.push_back((int) i); }
        for (size_t i = 0; i < fin.size(); ++i) if (fin[i].bin < 0) { items.push_back(&fin_feat[i]); item_of.push_back(-1 - (int) i); }
        const int NI = (int) items.size();
        if (R < 0.0 && live.size() >= 2) {
            // the novelty radius, once: radius x the median distance between the walkers at the first round
            const int nl = (int) std::min<size_t>(live.size(), 128);
            std::vector<std::pair<int, int>> pairs;
            for (int i = 0; i < nl; ++i) for (int j = i + 1; j < nl; ++j) pairs.push_back({i, j});
            std::vector<double> dd(pairs.size());
            parallel_for((int) pairs.size(), n_threads, [&](int k) { dd[k] = dist(live[pairs[k].first].feat, live[pairs[k].second].feat); });
            std::nth_element(dd.begin(), dd.begin() + dd.size() / 2, dd.end());
            R = p.radius * dd[dd.size() / 2];
            if (!(R > 0.0)) R = -1.0; // all identical so far: try again next round
        }
        std::vector<double> nd(NI, INFINITY);
        std::vector<int> nb(NI, -1);
        auto add_center_dists = [&](int c) {
            parallel_for(NI, n_threads, [&](int i) { const double d = dist(*items[i], centers[c]); if (d < nd[i]) { nd[i] = d; nb[i] = c; } });
        };
        for (int c = 0; c < (int) centers.size(); ++c) add_center_dists(c);
        if ((int) centers.size() >= p.max_bins) {
            // at the cap: retire centers no walker (live or finished) is nearest to, so new regions can still get bins
            std::vector<int> cnt(centers.size(), 0);
            for (int i = 0; i < NI; ++i) if (nb[i] >= 0) cnt[nb[i]]++;
            for (const auto & f : fin) if (f.bin >= 0) cnt[f.bin]++;
            std::vector<int> remap(centers.size(), -1);
            std::vector<std::vector<float>> c2;
            std::vector<int> b2;
            for (size_t c = 0; c < centers.size(); ++c) if (cnt[c] > 0) { remap[c] = (int) c2.size(); c2.push_back(std::move(centers[c])); b2.push_back(born[c]); }
            centers = std::move(c2);
            born = std::move(b2);
            for (auto & f : fin) if (f.bin >= 0) f.bin = remap[f.bin];
            for (auto & x : nb) if (x >= 0) x = remap[x];
        }
        int n_new = 0;
        while (NI > 0 && (int) centers.size() < p.max_bins) {
            const int imax = (int) (std::max_element(nd.begin(), nd.end()) - nd.begin());
            if (!centers.empty() && !(R > 0.0 && nd[imax] > R)) break;
            centers.push_back(*items[imax]);
            born.push_back(step);
            n_new++;
            add_center_dists((int) centers.size() - 1);
        }
        for (int i = 0; i < NI; ++i) {
            if (item_of[i] >= 0) live[item_of[i]].bin = nb[i]; else fin[-1 - item_of[i]].bin = nb[i];
        }
        double new_mass = 0.0;
        for (const auto & wk : live) if (n_new > 0 && wk.bin >= 0 && born[wk.bin] == step) new_mass += wk.w;

        if (resample_now && !live.empty()) {
            // an equal share of the population per occupied bin (the remainder to the heaviest bins)
            std::map<int, double> bmass;
            for (const auto & wk : live) bmass[wk.bin] += wk.w;
            std::vector<std::pair<double, int>> occ;
            for (const auto & kv : bmass) occ.push_back({kv.second, kv.first});
            std::sort(occ.begin(), occ.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            const int n_occ = (int) occ.size(), base = W / n_occ, rem = W % n_occ;
            std::map<int, int> target;
            for (int r = 0; r < n_occ; ++r) target[occ[r].second] = std::max(1, base + (r < rem ? 1 : 0));
            // merge: the two lightest walkers of a crowded bin become one, kept at random in proportion to the weights
            std::vector<char> dead(live.size(), 0);
            for (const auto & kv : target) {
                std::vector<int> mem_b;
                for (int i = 0; i < (int) live.size(); ++i) if (live[i].bin == kv.first) mem_b.push_back(i);
                while ((int) mem_b.size() > kv.second) {
                    std::sort(mem_b.begin(), mem_b.end(), [&](int a, int b) { return live[a].w < live[b].w; });
                    const int a = mem_b[0], b = mem_b[1];
                    const double wa = live[a].w, wb = live[b].w;
                    const bool keep_a = ud(rng) * (wa + wb) < wa;
                    const int k = keep_a ? a : b, g = keep_a ? b : a;
                    live[k].w = wa + wb;
                    dead[g] = 1;
                    release(live[g].seq);
                    mem_b.erase(std::find(mem_b.begin(), mem_b.end(), g));
                }
            }
            std::vector<walker> kept;
            for (size_t i = 0; i < live.size(); ++i) if (!dead[i]) kept.push_back(std::move(live[i]));
            live = std::move(kept);
            // split: the heaviest walker of a sparse bin becomes two with half the weight each
            for (int i = 0; i < (int) live.size(); ++i) live[i].fam = i;
            for (const auto & kv : target) {
                std::vector<int> mem_b;
                for (int i = 0; i < (int) live.size(); ++i) if (live[i].bin == kv.first) mem_b.push_back(i);
                while ((int) mem_b.size() < kv.second && !free_ids.empty()) {
                    const int a = *std::max_element(mem_b.begin(), mem_b.end(), [&](int x, int y) { return live[x].w < live[y].w; });
                    walker c = live[a];
                    c.seq = alloc();
                    llama_memory_seq_cp(mem, live[a].seq, c.seq, -1, -1);
                    live[a].w *= 0.5;
                    c.w = live[a].w;
                    live.push_back(std::move(c));
                    mem_b.push_back((int) live.size() - 1);
                }
            }
            // copies of one walker take stratified next tokens from its exact distribution
            std::map<int, std::vector<int>> fams;
            for (int i = 0; i < (int) live.size(); ++i) fams[live[i].fam].push_back(i);
            for (const auto & kv : fams) {
                if (kv.second.size() < 2) continue;
                const auto t = systematic(live[kv.second[0]].lp, (int) kv.second.size());
                for (size_t j = 0; j < kv.second.size(); ++j) live[kv.second[j]].forced = t[j];
            }
        }

        rd = server_explore_round();
        rd.step = step; rd.n_live = (int) live.size(); rd.n_done = (int) fin.size(); rd.n_new = n_new; rd.new_mass = new_mass;
        for (int c = 0; c < (int) centers.size(); ++c) {
            server_explore_bin b;
            b.id = c; b.born = born[c];
            std::vector<std::pair<double, const std::vector<int32_t> *>> m;
            for (const auto & wk : live) if (wk.bin == c) { b.mass += wk.w; b.n_walk++; m.push_back({wk.w, &wk.toks}); }
            for (const auto & f : fin) if (f.bin == c) { b.mass += f.w; m.push_back({f.w, &f.toks}); }
            if (m.empty()) continue;
            std::sort(m.begin(), m.end(), [](const auto & x, const auto & y) { return x.first > y.first; });
            for (int k = 0; k < std::min<int>(p.n_ex, (int) m.size()); ++k) { b.ex.push_back(*m[k].second); b.ex_w.push_back(m[k].first); }
            rd.bins.push_back(std::move(b));
        }
        std::sort(rd.bins.begin(), rd.bins.end(), [](const auto & x, const auto & y) { return x.mass > y.mass; });
        n_rounds++;
    };

    // ---- generation: one token per live walker per step, all in one batch ----
    long long n_tok = 0;
    int steps = 0;
    std::vector<int32_t> tok;
    std::vector<double> u;
    std::vector<float> ent;
    // fc: the next population from the children of the current one (sets each new walker's next token as forced)
    auto fc_select = [&]() {
        struct cand { int parent; int32_t tok; double q; };
        std::vector<std::vector<int32_t>> tops(live.size());
        std::vector<double> tail(live.size(), 0.0);
        // beam: each walker's top min(W, 64) first, then, against the W-th best candidate so far, every walker whose last
        // kept child still clears it gets all its children that do (later candidates only raise the bar, so this is exact)
        const int k0 = std::min(beam ? std::min(W, 64) : p.fc_top, n_vocab);
        parallel_for((int) live.size(), n_threads, [&](int i) {
            const auto & lp = live[i].lp;
            std::vector<int32_t> ord(n_vocab);
            std::iota(ord.begin(), ord.end(), 0);
            auto by_lp = [&](int32_t a, int32_t b) { return lp[a] > lp[b]; };
            std::nth_element(ord.begin(), ord.begin() + (k0 - 1), ord.end(), by_lp);
            std::sort(ord.begin(), ord.begin() + k0, by_lp);
            ord.resize(k0);
            double m = 0.0;
            for (int32_t x : ord) m += std::exp((double) lp[x]);
            tops[i] = std::move(ord);
            tail[i] = std::max(0.0, 1.0 - m);
        });
        if (beam && k0 < std::min(W, n_vocab)) {
            std::vector<double> qs;
            for (int i = 0; i < (int) live.size(); ++i) for (int32_t x : tops[i]) qs.push_back(live[i].w * std::exp((double) live[i].lp[x]));
            // fewer candidates than W (the first step, or few walkers): no bar yet, every walker gives its top W
            double bar = 0.0;
            if ((int) qs.size() >= W) {
                std::nth_element(qs.begin(), qs.begin() + (W - 1), qs.end(), std::greater<double>());
                bar = qs[W - 1];
            }
            const int kw = std::min(W, n_vocab);
            parallel_for((int) live.size(), n_threads, [&](int i) {
                const auto & lp = live[i].lp;
                if (!(live[i].w > 0.0) || live[i].w * std::exp((double) lp[tops[i].back()]) < bar) return;
                auto by_lp = [&](int32_t a, int32_t b) { return lp[a] > lp[b]; };
                std::vector<int32_t> more;
                if (bar > 0.0) {
                    const float lbar = (float) std::log(bar / live[i].w);
                    for (int v = 0; v < n_vocab; ++v) if (lp[v] >= lbar) more.push_back(v);
                    std::sort(more.begin(), more.end(), by_lp);
                    if ((int) more.size() > kw) more.resize(kw);
                } else {
                    more.resize(n_vocab);
                    std::iota(more.begin(), more.end(), 0);
                    std::nth_element(more.begin(), more.begin() + (kw - 1), more.end(), by_lp);
                    std::sort(more.begin(), more.begin() + kw, by_lp);
                    more.resize(kw);
                }
                if (more.size() > tops[i].size()) tops[i] = std::move(more);
            });
        }
        // a walker's tail (the tokens outside its top) is split into pieces of about 1 / walkers of the weight, each with
        // a token drawn from the tail by stratified sampling (one heavy tail walker would make the estimates noisy)
        std::vector<std::vector<std::pair<int32_t, double>>> tailc(live.size());
        std::vector<double> tu(live.size());
        for (auto & x : tu) x = ud(rng);
        parallel_for((int) live.size(), n_threads, [&](int i) {
            const double qt = live[i].w * tail[i];
            if (beam || !(qt > 0.0)) return; // beam: only the top children compete
            const int np = std::max(1, (int) std::ceil(qt * W));
            std::vector<char> in_top(n_vocab, 0);
            for (int32_t x : tops[i]) in_top[x] = 1;
            std::map<int32_t, int> cnt;
            double cum = 0.0;
            int k = 0;
            int32_t last = -1;
            for (int v = 0; v < n_vocab && k < np; ++v) {
                if (in_top[v]) continue;
                cum += std::exp((double) live[i].lp[v]);
                last = v;
                while (k < np && (k + tu[i]) / np * tail[i] < cum) { cnt[v]++; k++; }
            }
            if (k < np && last >= 0) cnt[last] += np - k;
            for (const auto & kv : cnt) tailc[i].push_back({kv.first, qt * kv.second / np});
        });
        std::vector<cand> cs;
        for (int i = 0; i < (int) live.size(); ++i) {
            for (int32_t x : tops[i]) cs.push_back({i, x, live[i].w * std::exp((double) live[i].lp[x])});
            for (const auto & tc : tailc[i]) cs.push_back({i, tc.first, tc.second});
        }
        std::sort(cs.begin(), cs.end(), [](const cand & a, const cand & b) { return a.q > b.q; });
        const int nc = (int) cs.size(), N = W;
        std::vector<double> suf(nc + 1, 0.0);
        for (int j = nc - 1; j >= 0; --j) suf[j] = suf[j + 1] + cs[j].q;
        // threshold c: keep the k heaviest, sum over the rest of q / c = N - k with every rest q < c
        int k = 0;
        double c = 0.0;
        if (nc <= N) { k = nc; } else {
            for (k = 0; k < N; ++k) { c = suf[k] / (N - k); if (cs[k].q < c) break; }
        }
        std::vector<std::pair<int, double>> pick; // candidate, new weight
        if (beam) k = std::min(nc, N);
        for (int j = 0; j < k; ++j) pick.push_back({j, cs[j].q});
        if (!beam && k < nc && k < N && c > 0.0) {
            double acc = 0.0, at = ud(rng) * c;
            for (int j = k; j < nc; ++j) {
                acc += cs[j].q;
                while (acc > at && (int) pick.size() < N) { pick.push_back({j, c}); at += c; } // rounding never adds one
            }
        }
        std::vector<int32_t> ptok(pick.size());
        for (size_t t = 0; t < pick.size(); ++t) ptok[t] = cs[pick[t].first].tok;
        // the new population: a parent's first child keeps its sequence, further children copy it, parents without
        // children are dropped
        std::vector<std::vector<int>> kids(live.size());
        for (size_t t = 0; t < pick.size(); ++t) kids[cs[pick[t].first].parent].push_back((int) t);
        last_npar = 0;
        for (const auto & kd : kids) last_npar += !kd.empty();
        for (int i = 0; i < (int) live.size(); ++i) if (kids[i].empty()) release(live[i].seq);
        std::vector<walker> next;
        for (int i = 0; i < (int) live.size(); ++i) {
            for (size_t j = 0; j < kids[i].size(); ++j) {
                const int t = kids[i][j];
                const float lpx = live[i].lp[ptok[t]];
                walker c2;
                if (j + 1 < kids[i].size()) {
                    // a copy: everything but the next-token distribution (the decode replaces it)
                    c2.toks = live[i].toks; c2.path = live[i].path; c2.feat = live[i].feat;
                    c2.bin = live[i].bin; c2.hot = live[i].hot; c2.fam = live[i].fam; c2.node = live[i].node;
                    c2.seq = alloc();
                    llama_memory_seq_cp(mem, live[i].seq, c2.seq, -1, -1);
                } else {
                    c2 = std::move(live[i]);
                }
                c2.w = pick[t].second;
                c2.forced = ptok[t];
                c2.path.push_back(lpx);
                c2.pushed = true;
                next.push_back(std::move(c2));
            }
        }
        live = std::move(next);
    };

    for (int step = 0; step < p.n_gen && !live.empty(); ++step) {
        const auto t_step = std::chrono::steady_clock::now();
        // stop before the cache could overflow: a step adds at most W cells
        if (live_cells() + W + 16 > N_CTX) { out.stopped = true; out.stop_reason = "memory"; break; }
        if (fc) fc_select();
        const int nl = (int) live.size();
        tok.assign(nl, 0); u.assign(nl, 0.0); ent.assign(nl, 0.0f);
        for (int i = 0; i < nl; ++i) u[i] = ud(rng);
        parallel_for(nl, n_threads, [&](int i) {
            walker & wk = live[i];
            if (wk.forced >= 0) { tok[i] = wk.forced; wk.forced = -1; if (!wk.pushed) wk.path.push_back(wk.lp[tok[i]]); wk.pushed = false; return; }
            // proposal q = p^alpha / Z (alpha = 1: p itself); the weight carries p / q
            double lz = 0.0;
            if (p.alpha != 1.0f) {
                double m = -INFINITY;
                for (int v = 0; v < n_vocab; ++v) m = std::max(m, (double) p.alpha * wk.lp[v]);
                double z = 0.0;
                for (int v = 0; v < n_vocab; ++v) z += std::exp(p.alpha * wk.lp[v] - m);
                lz = m + std::log(z);
            }
            double cum = 0.0;
            int x = n_vocab - 1;
            for (int v = 0; v < n_vocab; ++v) {
                cum += std::exp(p.alpha == 1.0f ? (double) wk.lp[v] : p.alpha * wk.lp[v] - lz);
                if (cum >= u[i]) { x = v; break; }
            }
            if (p.alpha != 1.0f) wk.w *= std::exp((double) wk.lp[x] - (p.alpha * wk.lp[x] - lz));
            tok[i] = x;
            wk.path.push_back(wk.lp[x]);
        });
        for (int i = 0; i < nl; ++i) {
            batch.token[i] = tok[i]; batch.pos[i] = s + 1 + step; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = live[i].seq; batch.logits[i] = 1;
        }
        batch.n_tokens = nl;
        cap.rows = 0;
        cap.enabled = true;
        const int rc = llama_decode(ctx, batch);
        cap.enabled = false;
        if (rc == 1) {
            // no room for the batch after all (scattered free cells): keep the chosen tokens, undecoded, and stop
            for (int i = 0; i < nl; ++i) {
                walker & wk = live[i];
                wk.toks.push_back(tok[i]);
                npar.push_back(wk.node);
                if (p.history) out.nodes.push_back({wk.node, tok[i], step, wk.w, wk.path.back()});
                wk.node = (int) npar.size() - 1;
            }
            steps = step + 1;
            out.stopped = true;
            out.stop_reason = "memory";
            break;
        }
        if (rc != 0) { err = "explore: decode failed (step " + std::to_string(step) + ")"; return false; }
        if (cap.rows != nl) { err = "explore: l_out capture failed (step " + std::to_string(step) + ")"; return false; }
        n_tok += nl;
        steps = step + 1;
        parallel_for(nl, n_threads, [&](int i) {
            walker & wk = live[i];
            wk.lp.resize(n_vocab); // fc/beam copies start without one
            log_softmax(llama_get_logits_ith(ctx, i), wk.lp.data(), n_vocab, 1.0f);
            double H = 0.0;
            for (int v = 0; v < n_vocab; ++v) H -= std::exp((double) wk.lp[v]) * wk.lp[v];
            ent[i] = (float) H;
            const float * h = cap.buf.data() + (size_t) i * n_embd;
            for (int d = 0; d < n_embd; ++d) wk.feat[d] = (1.0f - p.ema) * wk.feat[d] + p.ema * h[d];
            wk.toks.push_back(tok[i]);
        });
        // finished walkers leave the population: end of document, or a humane endpoint (loop, incoherence)
        std::vector<walker> kept;
        for (int i = 0; i < nl; ++i) {
            walker & wk = live[i];
            npar.push_back(wk.node);
            if (p.history) out.nodes.push_back({wk.node, tok[i], step, wk.w, wk.path.back()});
            wk.node = (int) npar.size() - 1;
            wk.hot = ent[i] > INCOH_H ? wk.hot + 1 : 0;
            const int end = llama_vocab_is_eog(vocab, tok[i]) || llama_vocab_is_control(vocab, tok[i]) ? 1 : find_loop(wk.toks, LOOP_MAX_P, LOOP_MIN_SPAN) > 0 ? 2 : wk.hot >= INCOH_RUN ? 3 : 0;
            if (end) {
                fin.push_back({wk.toks, wk.path, wk.w, -1, end, wk.node});
                fin_feat.push_back(wk.feat);
                release(wk.seq);
            } else {
                kept.push_back(std::move(wk));
            }
        }
        live = std::move(kept);
        if (!live.empty() && (step + 1) % p.tau == 0 && step + 1 < p.n_gen) {
            server_explore_round rd;
            do_round(step + 1, !fc && !plain, rd);
            rd.cells = live_cells();
            if (p.on_round && !p.on_round(rd)) break;
        }
        if (g_explore_stop.exchange(false)) { out.stopped = true; out.stop_reason = "user"; break; }
        // pacing for long runs: rest (1 / duty - 1) x this step's time, so the GPU cools
        const auto now = std::chrono::steady_clock::now();
        if (p.duty < 1.0f && now - t_start > std::chrono::seconds(5)) {
            std::this_thread::sleep_for(std::chrono::duration<double>(std::chrono::duration<double>(now - t_step).count() * (1.0 / p.duty - 1.0)));
        }
    }
    do_round(steps, false, out.last);
    out.last.cells = live_cells();
    for (const auto & wk : live) out.walkers.push_back({wk.toks, wk.path, wk.w, wk.bin, 0, wk.node});
    for (const auto & f : fin) out.walkers.push_back(f);
    out.radius   = R;
    out.n_rounds = n_rounds;
    out.n_tok    = n_tok;
    out.sec      = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    // ---- exposure log: plain sampling (proposal p^alpha), no push; counted like rollouts ----
    int n_end = 0;
    for (const auto & f : fin) n_end += f.done >= 2;
    char mode[256];
    snprintf(mode, sizeof(mode), "%d walkers, %s, features L%d, proposal p^%.2f", W, fc ? "optimal resampling of the top children every token" :
             plain ? "no resampling (plain rollouts)" : ("split/merge between bins every " + std::to_string(p.tau) + " tokens").c_str(), p.layer, p.alpha);
    if (!p.log_group.empty()) {
        steer_group & g = g_steer.grp;
        const auto now = std::chrono::steady_clock::now();
        if (g.n_req > 0 && (g.kind != "explore" || g.name != p.log_group || g.mode != mode)) flush_group();
        if (g.n_req == 0) { g.kind = "explore"; g.name = p.log_group; g.mode = mode; g.model = g_steer.model_name; g.note = p.note; g.t0 = now; }
        g.tok += n_tok; g.n_ctx += (int) out.walkers.size(); g.n_end += n_end; g.n_req++;
        if (p.log_flush || g.n_req >= 100 || now - g.t0 > std::chrono::minutes(10)) flush_group();
    } else {
        char row[2048];
        snprintf(row, sizeof(row), "%s,loom-explore,%s,%s,%.3f,%lld,%d,%s,%.3f,%s\n", today().c_str(), csv_q(g_steer.model_name).c_str(),
                 csv_q(std::string("exploration with weighted walkers, no push: ancestral sampling with unbiased resampling of the walkers; ") + mode).c_str(),
                 0.0, n_tok, (int) out.walkers.size(), csv_q(ENDPOINT_DESC).c_str(), out.walkers.empty() ? 0.0 : (double) n_end / out.walkers.size(),
                 csv_q("llama-server /steer (explore)" + (p.note.empty() ? "" : "; " + p.note)).c_str());
        write_log_row(row);
    }
    return true;
}
