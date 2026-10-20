#include "server-steer.h"
#include "server-jlens.h"

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <mutex>
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
};

bool steer_cb_eval(ggml_tensor * t, bool ask, void * ud) {
    auto * c = (steer_capture *) ud;
    if (!c->enabled || c->name != t->name) return !ask;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 || t->ne[0] != c->n_embd || !ggml_is_contiguous(t)) {
        c->rows = -1; // checked by the caller
        return true;
    }
    c->rows = ggml_nrows(t);
    c->buf.resize((size_t) c->rows * c->n_embd);
    ggml_backend_tensor_get(t, c->buf.data(), 0, ggml_nbytes(t));
    return true;
}

// exposure counts of a log group, written as one row
struct steer_group {
    std::string name, mode, model, note;
    long long   tok = 0;
    int         n_ctx = 0, n_end = 0, n_req = 0;
    float       max_dose = 0.0f, h_lo = 0.0f, h_hi = 0.0f;
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

constexpr int N_CTX     = 8192;
constexpr int N_BATCH   = 512;
constexpr int N_UBATCH  = 256; // l_out capture sees one ubatch per callback: captured decodes stay within it
constexpr int N_SEQ_MAX = 64; // the last one holds the shared unsteered prefix
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
    char manip[768];
    snprintf(manip, sizeof(manip), "steer along %zu layer-sweep probe directions (%s) x dose x mean|h| (%.1f-%.1f); %s", g.dirs.size(),
             g.dirs.size() <= 4 ? [&] { std::string o; for (auto & d : g.dirs) o += (o.empty() ? "" : " ") + d; return o; }().c_str() : "blocks of each",
             g.h_lo, g.h_hi, g.mode.c_str());
    char row[2048];
    snprintf(row, sizeof(row), "%s,loom-steer,%s,%s,%.3f,%lld,%d,%s,%.3f,%s\n", today().c_str(), csv_q(g.model).c_str(), csv_q(manip).c_str(),
             g.max_dose, g.tok, g.n_ctx, csv_q(ENDPOINT_DESC).c_str(), g.n_ctx ? (double) g.n_end / g.n_ctx : 0.0,
             csv_q("llama-server /steer, log group '" + g.name + "': " + std::to_string(g.n_req) + " requests" + (g.note.empty() ? "" : "; " + g.note)).c_str());
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
    if (NS < 1 || G < 1 || G > 512 || n_arm * NS > N_SEQ_MAX - 1) {
        err = "steer: need 1 <= n_gen <= 512 and (arms x n_seq) <= " + std::to_string(N_SEQ_MAX - 1);
        return false;
    }
    if (!(p.temp > 0.0f)) { err = "steer: temperature must be > 0"; return false; }
    if (s + n_arm * NS * (G + 1) > N_CTX || s + n_arm * (n_f + 1) > N_CTX) {
        err = "steer: text too long for the steering context (" + std::to_string(N_CTX) + " cells): shorten the text, n_seq or n_gen";
        return false;
    }

    if (!g_steer.ctx) {
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
    }
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
    for (int a = 0; a < n_arm; ++a) {
        server_steer_arm & arm = out.arms[a];
        arm.dose = doses[a];
        arm.push = doses[a] * h_norm;
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
                hot[k] = sq.ent.back() > INCOH_H ? hot[k] + 1 : 0;
                if (hot[k] >= INCOH_RUN) { sq.endpoint = 2; continue; }
                if (find_loop(sq.tok, LOOP_MAX_P, LOOP_MIN_SPAN) > 0) { sq.endpoint = 1; continue; }
                if (llama_vocab_is_eog(vocab, sq.tok.back())) continue; // end of document: stop, not an endpoint
                nxt.push_back(k);
            }
            act = nxt;
            for (int j = 0; j < (int) act.size(); ++j) {
                const int k = act[j];
                batch.token[j] = arm.seqs[k].tok.back(); batch.pos[j] = s + 1 + step; batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = a * NS + k; batch.logits[j] = 1;
            }
            batch.n_tokens = (int) act.size();
        }
        for (int k = 0; k < NS; ++k) llama_memory_seq_rm(mem, a * NS + k, -1, -1);
        if (a > 0) {
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
            if (g.n_req > 0 && (g.name != p.log_group || g.mode != mode)) flush_group();
            if (g.n_req == 0) { g.name = p.log_group; g.mode = mode; g.model = g_steer.model_name; g.note = p.note; g.t0 = now; g.h_lo = g.h_hi = h_norm; }
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
