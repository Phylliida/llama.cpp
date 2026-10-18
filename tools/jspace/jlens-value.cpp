#include "jlens.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

// exact value lens and branch gradients over a whole text: see jlens_value_exact / jlens_branch_grad in jlens.h

namespace {

double now_sec() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// point a leaf at existing device memory (gallocr skips tensors that already have data)
void wire(ggml_tensor * leaf, const ggml_tensor * src, size_t offset) {
    leaf->data   = (char *) src->data + offset;
    leaf->buffer = src->buffer;
}

// device copies of the forward pass (block outputs that are kept, keys/values of every block) and the
// key/value gradients carried back from later chunks
struct value_store {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> hl;     // [n_layer + 1]: hl[il + 1] = output of block il, hl[0] = token embeddings; nullptr if not kept
    std::vector<ggml_tensor *> k, v;   // per block, [D, Hkv, N]
    std::vector<ggml_tensor *> dk, dv; // per block in the backward range, else nullptr
    ggml_tensor * g    = nullptr;      // [n_embd, N] gradient injected at the top of a layer group
    ggml_tensor * seed = nullptr;      // [1] = 1.0

    ggml_tensor * h(int il) const { return hl[il + 1]; }

    // keep: block outputs to keep (-1 = embeddings); backward blocks lo+1 .. n_layer-1 get gradient stores
    bool alloc(const jlens_weights & m, ggml_backend_buffer_type_t buft, int N, const std::vector<int> & keep, int lo, bool need_g) {
        const auto & hp = m.hparams;
        const int64_t D = hp.n_embd_head, Hkv = hp.n_head_kv;
        ggml_init_params ip = { (size_t) (4 * hp.n_layer + hp.n_layer + 1 + 2) * ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        hl.assign(hp.n_layer + 1, nullptr);
        for (int il : keep) hl[il + 1] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, N);
        k.assign(hp.n_layer, nullptr);
        v.assign(hp.n_layer, nullptr);
        dk.assign(hp.n_layer, nullptr);
        dv.assign(hp.n_layer, nullptr);
        for (int il = 0; il < hp.n_layer; ++il) {
            k[il] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, N);
            v[il] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, N);
            if (il > lo) {
                dk[il] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, N);
                dv[il] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hkv, N);
            }
        }
        if (need_g) g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, N);
        seed = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) return false;
        ggml_backend_buffer_clear(buf, 0);
        const float one = 1.0f;
        ggml_backend_tensor_set(seed, &one, 0, sizeof(float));
        return true;
    }

    ~value_store() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

void causal_mask(int64_t a, int c, int64_t n_kv, std::vector<float> & mask) {
    mask.assign((size_t) n_kv * c, 0.0f);
    for (int i = 0; i < c; ++i)
        for (int64_t j = a + i + 1; j < n_kv; ++j)
            mask[(size_t) j + (size_t) i * n_kv] = -FLT_MAX;
}

// forward sweep in chunks with explicit attention: fills st.k, st.v (every block) and the kept block outputs.
// emb_add (optional, host [N][n_embd]) is added to the token embeddings.
bool forward_chunks(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                    const std::vector<int32_t> & tokens, int chunk, value_store & st, const float * emb_add,
                    bool & oom, std::string & err) {
    const auto & hp = m.hparams;
    const int N = (int) tokens.size();
    const int64_t D = hp.n_embd_head, Hkv = hp.n_head_kv;
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    std::vector<int32_t> posv;
    std::vector<float> mask;
    for (int64_t a = 0; a < N; a += chunk) {
        const int64_t b = std::min<int64_t>(N, a + chunk);
        const int c = (int) (b - a);
        jlens_chunk_forward cf;
        if (!jlens_build_chunk_forward(m, a, c, b, cf, emb_add != nullptr)) {
            ggml_gallocr_free(galloc);
            err = "value: forward chunk build failed";
            return false;
        }
        wire(cf.one, st.seed, 0);
        for (int il = 0; il < hp.n_layer; ++il) {
            wire(cf.k_kv[il], st.k[il], 0);
            wire(cf.v_kv[il], st.v[il], 0);
            for (int kv = 0; kv < 2; ++kv) {
                ggml_tensor * dst = ggml_new_tensor_3d(cf.gctx, GGML_TYPE_F32, D, Hkv, c);
                wire(dst, kv == 0 ? st.k[il] : st.v[il], (size_t) a * st.k[il]->nb[2]);
                ggml_build_forward_expand(cf.gf, ggml_cpy(cf.gctx, kv == 0 ? cf.k_live[il] : cf.v_live[il], dst));
            }
        }
        for (int il = -1; il < hp.n_layer; ++il) {
            if (!st.h(il)) continue;
            ggml_tensor * hdst = ggml_new_tensor_2d(cf.gctx, GGML_TYPE_F32, hp.n_embd, c);
            wire(hdst, st.h(il), (size_t) a * st.h(il)->nb[1]);
            ggml_build_forward_expand(cf.gf, ggml_cpy(cf.gctx, il < 0 ? cf.emb : cf.l_out[il], hdst));
        }
        if (!ggml_gallocr_alloc_graph(galloc, cf.gf)) {
            ggml_free(cf.gctx);
            ggml_gallocr_free(galloc);
            oom = true;
            err = "value: forward chunk alloc failed (out of VRAM?)";
            return false;
        }
        posv.resize(c);
        for (int i = 0; i < c; ++i) posv[i] = (int32_t) (a + i);
        causal_mask(a, c, b, mask);
        ggml_backend_tensor_set(cf.tokens, tokens.data() + a, 0, c * sizeof(int32_t));
        ggml_backend_tensor_set(cf.pos,    posv.data(),       0, c * sizeof(int32_t));
        ggml_backend_tensor_set(cf.mask,   mask.data(),       0, mask.size() * sizeof(float));
        if (emb_add) {
            ggml_backend_tensor_set(cf.emb_add, emb_add + (size_t) a * hp.n_embd, 0, (size_t) c * hp.n_embd * sizeof(float));
        }
        const bool ok = ggml_backend_graph_compute(backend, cf.gf) == GGML_STATUS_SUCCESS;
        ggml_free(cf.gctx);
        if (!ok) {
            ggml_gallocr_free(galloc);
            err = "value: forward chunk compute failed";
            return false;
        }
    }
    ggml_gallocr_free(galloc);
    return true;
}

// head over the final block output (kept in st): per position the log-likelihood of the next token, and the
// top-k candidates with their probabilities (cand/prob may be null)
bool head_pass(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
               const std::vector<int32_t> & tokens, const value_store & st, int topk, int n_real_in,
               std::vector<double> & ll, std::vector<int32_t> * cand, std::vector<float> * prob, std::string & err) {
    const auto & hp = m.hparams;
    const int N = (int) tokens.size();
    ggml_tensor * hfin = st.h((int) hp.n_layer - 1);
    ggml_tensor * out_norm = m.get("output_norm.weight");
    ggml_tensor * out_w    = m.get("output.weight");
    const int64_t n_vocab = out_w->ne[1];
    const int n_real = n_real_in > 0 ? (int) std::min<int64_t>(n_real_in, n_vocab) : (int) n_vocab; // logits may be padded
    ll.assign(N, NAN);
    if (cand) cand->assign((size_t) N * topk, -1);
    if (prob) prob->assign((size_t) N * topk, 0.0f);
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    const int chunk = 256;
    const int nt = (int) std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    std::vector<float> lg;
    for (int p0 = 0; p0 < N; p0 += chunk) {
        const int np = std::min(chunk, N - p0);
        ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * av = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.n_embd, np);
        wire(av, hfin, (size_t) p0 * hfin->nb[1]);
        ggml_tensor * logits = ggml_mul_mat(c, out_w, ggml_mul(c, ggml_rms_norm(c, av, hp.norm_eps), out_norm));
        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, logits);
        const bool ok = ggml_gallocr_alloc_graph(galloc, gf) && ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            lg.resize((size_t) n_vocab * np);
            ggml_backend_tensor_get(logits, lg.data(), 0, lg.size() * sizeof(float));
        }
        ggml_free(c);
        if (!ok) {
            ggml_gallocr_free(galloc);
            err = "value: head compute failed";
            return false;
        }
        std::vector<std::thread> ts;
        const int per = (np + nt - 1) / nt;
        for (int t = 0; t < nt; ++t) {
            const int q0 = t * per, q1 = std::min(np, q0 + per);
            if (q0 >= q1) break;
            ts.emplace_back([&, q0, q1] {
                std::vector<int> idx(n_real);
                for (int q = q0; q < q1; ++q) {
                    const float * z = lg.data() + (size_t) q * n_vocab;
                    float mx = -INFINITY;
                    for (int x = 0; x < n_real; ++x) mx = std::max(mx, z[x]);
                    double se = 0.0;
                    for (int x = 0; x < n_real; ++x) se += std::exp((double) z[x] - mx);
                    const double lse = mx + std::log(se);
                    const int s = p0 + q;
                    if (s + 1 < N) ll[s] = (double) z[tokens[s + 1]] - lse;
                    if (cand) {
                        for (int x = 0; x < n_real; ++x) idx[x] = x;
                        std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](int i1, int i2) { return z[i1] > z[i2]; });
                        for (int r = 0; r < topk; ++r) {
                            (*cand)[(size_t) s * topk + r] = idx[r];
                            if (prob) (*prob)[(size_t) s * topk + r] = (float) std::exp((double) z[idx[r]] - lse);
                        }
                    }
                }
            });
        }
        for (auto & t : ts) t.join();
    }
    ggml_gallocr_free(galloc);
    return true;
}

// one backward sweep for the blocks slot_layer+1 .. top_layer, from the last chunk to the first (main mode) or
// over independent chunks (now mode). top_layer = last block: the loss is the next-token NLL; else the loss
// gradient at the top_layer output comes from st.g. grad_h receives d loss / d h at the slot_layer output.
bool sweep(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
           const std::vector<int32_t> & tokens, int slot_layer, int top_layer, int chunk, bool now_mode,
           value_store & st, std::vector<float> & grad_h, bool & oom, std::string & err) {
    const auto & hp = m.hparams;
    const int N = (int) tokens.size();
    const int64_t n_embd = hp.n_embd, D = hp.n_embd_head, Hkv = hp.n_head_kv;
    const bool head = top_layer == hp.n_layer - 1;
    const int n_up = top_layer - slot_layer;
    const int n_chunks = (N + chunk - 1) / chunk;
    grad_h.assign((size_t) N * n_embd, 0.0f);

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    std::vector<int32_t> posv;
    std::vector<float> mask, labels;
    for (int jj = 0; jj < n_chunks; ++jj) {
        const int j = n_chunks - 1 - jj;
        const int64_t a = (int64_t) j * chunk;
        const int64_t b = std::min<int64_t>(N, a + chunk);
        const int c = (int) (b - a);
        const int n_rows = (int) std::min<int64_t>(c, N - 1 - a); // positions that have a next token
        if (head && n_rows < 1) continue; // only the final position: no loss, nothing to send back

        jlens_upper_chunk ch;
        if (!jlens_build_upper_chunk(m, slot_layer, a, c, b, head ? n_rows : c, now_mode, ch, top_layer)) {
            ggml_gallocr_free(galloc);
            err = "value: chunk graph build failed";
            return false;
        }
        wire(ch.h, st.h(slot_layer), (size_t) a * st.h(slot_layer)->nb[1]);
        wire(ch.one, st.seed, 0);
        if (!head) wire(ch.g_top, st.g, (size_t) a * st.g->nb[1]);
        for (int u = 0; u < n_up; ++u) {
            const int il = slot_layer + 1 + u;
            wire(ch.k_kv[u], st.k[il], 0);
            wire(ch.v_kv[u], st.v[il], 0);
            if (!now_mode) {
                wire(ch.dk_in[u], st.dk[il], (size_t) a * st.dk[il]->nb[2]);
                wire(ch.dv_in[u], st.dv[il], (size_t) a * st.dv[il]->nb[2]);
            }
        }

        std::vector<ggml_tensor *> grad_accs(ggml_graph_n_nodes(ch.gf), nullptr);
        for (int i = 0; i < ggml_graph_n_nodes(ch.gf); ++i) {
            if (ggml_graph_node(ch.gf, i) == ch.loss) grad_accs[i] = st.seed;
        }
        ggml_cgraph * gb = ggml_graph_dup(ch.gctx, ch.gf, /*force_grads=*/true);
        ggml_build_backward_expand(ch.gctx, gb, grad_accs.data());
        ggml_tensor * gh = ggml_graph_get_grad(gb, ch.h);
        if (!gh) {
            ggml_free(ch.gctx);
            ggml_gallocr_free(galloc);
            err = "value: no gradient for the slot input";
            return false;
        }
        ggml_set_output(gh);
        if (!now_mode && a > 0) {
            // carry the gradient for the earlier positions' keys/values back: dk[0, a) += grad k_kv[0, a)
            for (int u = 0; u < n_up; ++u) {
                const int il = slot_layer + 1 + u;
                for (int kv = 0; kv < 2; ++kv) {
                    ggml_tensor * g = ggml_graph_get_grad(gb, kv == 0 ? ch.k_kv[u] : ch.v_kv[u]);
                    if (!g) {
                        ggml_free(ch.gctx);
                        ggml_gallocr_free(galloc);
                        err = "value: no gradient for the earlier keys/values";
                        return false;
                    }
                    ggml_tensor * dst = ggml_new_tensor_3d(ch.gctx, GGML_TYPE_F32, D, Hkv, a);
                    wire(dst, kv == 0 ? st.dk[il] : st.dv[il], 0);
                    ggml_tensor * src = ggml_view_3d(ch.gctx, g, D, Hkv, a, g->nb[1], g->nb[2], 0);
                    ggml_build_forward_expand(gb, ggml_add_inplace(ch.gctx, dst, src));
                }
            }
        }
        if (!ggml_gallocr_alloc_graph(galloc, gb)) {
            ggml_free(ch.gctx);
            ggml_gallocr_free(galloc);
            oom = true;
            err = "value: chunk graph alloc failed (out of VRAM?)";
            return false;
        }
        posv.resize(c);
        for (int i = 0; i < c; ++i) posv[i] = (int32_t) (a + i);
        causal_mask(a, c, b, mask);
        ggml_backend_tensor_set(ch.pos,  posv.data(), 0, posv.size() * sizeof(int32_t));
        ggml_backend_tensor_set(ch.mask, mask.data(), 0, mask.size() * sizeof(float));
        if (head) {
            const int64_t n_vocab = ch.labels->ne[0];
            labels.assign((size_t) n_vocab * n_rows, 0.0f);
            for (int i = 0; i < n_rows; ++i) labels[(size_t) i * n_vocab + tokens[a + i + 1]] = 1.0f;
            ggml_backend_tensor_set(ch.labels, labels.data(), 0, labels.size() * sizeof(float));
        }
        if (ggml_backend_graph_compute(backend, gb) != GGML_STATUS_SUCCESS) {
            ggml_free(ch.gctx);
            ggml_gallocr_free(galloc);
            err = "value: chunk compute failed";
            return false;
        }
        ggml_backend_tensor_get(gh, grad_h.data() + (size_t) a * n_embd, 0, (size_t) c * n_embd * sizeof(float));
        ggml_free(ch.gctx);
    }
    ggml_gallocr_free(galloc);
    return true;
}

// unchunked forward (explicit attention, O(N^2) memory): reference for checks
bool forward_full(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                  const std::vector<int32_t> & tokens, int slot_layer, value_store & st, bool & oom, std::string & err) {
    const auto & hp = m.hparams;
    const int N = (int) tokens.size();
    jlens_forward fwd;
    if (!jlens_build_forward(m, N, /*with_grad_flags=*/false, fwd, -1, /*use_flash_attn=*/false, /*with_head=*/false)) {
        err = "value: forward graph build failed";
        return false;
    }
    auto keep = [](ggml_tensor * t) {
        ggml_set_output(t);
        if (t->view_src) ggml_set_output(t->view_src);
    };
    keep(fwd.l_out[slot_layer]);
    for (int il = slot_layer + 1; il < hp.n_layer; ++il) {
        keep(fwd.k_out[il]);
        keep(fwd.v_out[il]);
    }
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(galloc, fwd.gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(fwd.gctx);
        oom = true;
        err = "value: forward graph alloc failed (out of VRAM?)";
        return false;
    }
    std::vector<int32_t> posv(N);
    for (int i = 0; i < N; ++i) posv[i] = i;
    std::vector<float> mask;
    causal_mask(0, N, N, mask);
    ggml_backend_tensor_set(fwd.tokens, tokens.data(), 0, N * sizeof(int32_t));
    ggml_backend_tensor_set(fwd.pos,    posv.data(),   0, N * sizeof(int32_t));
    ggml_backend_tensor_set(fwd.mask,   mask.data(),   0, mask.size() * sizeof(float));
    const bool ok = ggml_backend_graph_compute(backend, fwd.gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_copy(fwd.l_out[slot_layer], st.h(slot_layer));
        for (int il = slot_layer + 1; il < hp.n_layer; ++il) {
            ggml_backend_tensor_copy(fwd.k_out[il], st.k[il]);
            ggml_backend_tensor_copy(fwd.v_out[il], st.v[il]);
        }
    }
    ggml_gallocr_free(galloc);
    ggml_free(fwd.gctx);
    if (!ok) err = "value: forward compute failed";
    return ok;
}

bool value_exact_once(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                      const std::vector<int32_t> & tokens, int slot_layer, int chunk, bool keep_vectors, bool full_fwd,
                      jlens_value_result & out, bool & oom, std::string & err) {
    const auto & hp = m.hparams;
    const int N = (int) tokens.size();
    const int64_t n_embd = hp.n_embd;
    out = jlens_value_result();
    out.chunk = chunk;
    out.fut_norm.assign(N, 0.0f);
    out.now_norm.assign(N, 0.0f);

    value_store st;
    if (!st.alloc(m, buft, N, { slot_layer }, slot_layer, /*need_g=*/false)) {
        oom = true;
        err = "value: store alloc failed (out of VRAM?)";
        return false;
    }
    double t0 = now_sec();
    const bool fwd_ok = full_fwd ? forward_full(m, backend, buft, tokens, slot_layer, st, oom, err)
                                 : forward_chunks(m, backend, buft, tokens, chunk, st, nullptr, oom, err);
    if (!fwd_ok) return false;
    double t1 = now_sec();
    out.sec_fwd = t1 - t0;
    if (keep_vectors) {
        out.h.resize((size_t) N * n_embd);
        ggml_backend_tensor_get(st.h(slot_layer), out.h.data(), 0, out.h.size() * sizeof(float));
    }

    // incl[s] = d (sum of NLL at all positions >= s) / d h_s, now[s] = d NLL_s / d h_s
    const int top = (int) hp.n_layer - 1;
    std::vector<float> incl, now;
    if (!sweep(m, backend, buft, tokens, slot_layer, top, chunk, /*now_mode=*/false, st, incl, oom, err)) return false;
    double t2 = now_sec();
    out.sec_main = t2 - t1;
    if (!sweep(m, backend, buft, tokens, slot_layer, top, chunk, /*now_mode=*/true, st, now, oom, err)) return false;
    out.sec_now = now_sec() - t2;

    // NLL gradients -> log-likelihood gradients (sign flip); fut = incl - now
    if (keep_vectors) {
        out.fut.assign((size_t) N * n_embd, 0.0f);
        out.now.assign((size_t) N * n_embd, 0.0f);
    }
    for (int s = 0; s < N; ++s) {
        double nf = 0.0, nn = 0.0;
        for (int64_t e = 0; e < n_embd; ++e) {
            const size_t i = (size_t) s * n_embd + e;
            const float f = -(incl[i] - now[i]), g = -now[i];
            nf += (double) f * f;
            nn += (double) g * g;
            if (keep_vectors) {
                out.fut[i] = f;
                out.now[i] = g;
            }
        }
        out.fut_norm[s] = (float) std::sqrt(nf);
        out.now_norm[s] = (float) std::sqrt(nn);
    }
    return true;
}

// candidate embedding dot products on the GPU: dots[s][r] = g[s] . E[ids[s][r]] for r < n_ids
bool embed_dots(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                const std::vector<int32_t> & ids, int n_ids, const std::vector<float> & g, int N,
                std::vector<float> & dots, std::string & err) {
    const int64_t n_embd = m.hparams.n_embd;
    dots.assign((size_t) N * n_ids, 0.0f);
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    const int chunk = 128;
    for (int p0 = 0; p0 < N; p0 += chunk) {
        const int np = std::min(chunk, N - p0);
        ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * tid = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) n_ids * np);
        ggml_set_input(tid);
        ggml_tensor * tg = ggml_new_tensor_3d(c, GGML_TYPE_F32, n_embd, 1, np);
        ggml_set_input(tg);
        ggml_tensor * rows = ggml_reshape_3d(c, ggml_get_rows(c, m.get("token_embd.weight"), tid), n_embd, n_ids, np);
        ggml_tensor * d = ggml_sum_rows(c, ggml_mul(c, rows, tg)); // [1, n_ids, np]
        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, d);
        bool ok = ggml_gallocr_alloc_graph(galloc, gf);
        if (ok) {
            ggml_backend_tensor_set(tid, ids.data() + (size_t) p0 * n_ids, 0, (size_t) n_ids * np * sizeof(int32_t));
            ggml_backend_tensor_set(tg, g.data() + (size_t) p0 * n_embd, 0, (size_t) np * n_embd * sizeof(float));
            ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        }
        if (ok) ggml_backend_tensor_get(d, dots.data() + (size_t) p0 * n_ids, 0, (size_t) n_ids * np * sizeof(float));
        ggml_free(c);
        if (!ok) {
            ggml_gallocr_free(galloc);
            err = "value: candidate dot products failed";
            return false;
        }
    }
    ggml_gallocr_free(galloc);
    return true;
}

bool branch_grad_once(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                      const std::vector<int32_t> & tokens, int topk, int chunk, int n_real, jlens_branch_result & out,
                      std::vector<float> * grad_emb, int max_group, bool & oom, std::string & err) {
    const auto & hp = m.hparams;
    const int N = (int) tokens.size();
    const int64_t n_embd = hp.n_embd;
    const int L = (int) hp.n_layer;
    out = jlens_branch_result();
    out.k = topk;
    out.chunk = chunk;

    // layer groups of at most 21 blocks, top down: (lo, hi] with the head on the first
    std::vector<int> bounds = { L - 1 };
    while (bounds.back() > -1) bounds.push_back(std::max(-1, bounds.back() - max_group));
    std::vector<int> keep(bounds.begin(), bounds.end()); // block outputs at the group boundaries, incl. the last block and -1

    value_store st;
    if (!st.alloc(m, buft, N, keep, -1, /*need_g=*/true)) {
        oom = true;
        err = "value: store alloc failed (out of VRAM?)";
        return false;
    }
    double t0 = now_sec();
    if (!forward_chunks(m, backend, buft, tokens, chunk, st, nullptr, oom, err)) return false;
    std::vector<double> ll;
    if (!head_pass(m, backend, buft, tokens, st, topk, n_real, ll, &out.cand, &out.prob, err)) return false;
    double t1 = now_sec();
    out.sec_fwd = t1 - t0;

    // chain the exact gradient of the total NLL down the stack, one layer group per sweep
    std::vector<float> grad;
    for (size_t gi = 0; gi + 1 < bounds.size(); ++gi) {
        const int hi = bounds[gi], lo = bounds[gi + 1];
        if (gi > 0) ggml_backend_tensor_set(st.g, grad.data(), 0, grad.size() * sizeof(float));
        if (!sweep(m, backend, buft, tokens, lo, hi, chunk, /*now_mode=*/false, st, grad, oom, err)) return false;
    }
    out.sec_bwd = now_sec() - t1;

    // grad[s] = d (sum of NLL at all positions >= s) / d emb_s. The token x_{s+1} enters only through emb_{s+1},
    // so the linearized change in the later log-likelihood if x_{s+1} were candidate c is
    //   delta = -grad[s+1] . (E[c] - E[x_{s+1}])
    const int nk = topk + 1; // candidates + the actual token
    std::vector<int32_t> ids((size_t) N * nk, 0);
    std::vector<float> gnext((size_t) N * n_embd, 0.0f);
    for (int s = 0; s + 1 < N; ++s) {
        for (int r = 0; r < topk; ++r) ids[(size_t) s * nk + r] = out.cand[(size_t) s * topk + r];
        ids[(size_t) s * nk + topk] = tokens[s + 1];
        std::copy(grad.begin() + (size_t) (s + 1) * n_embd, grad.begin() + (size_t) (s + 2) * n_embd, gnext.begin() + (size_t) s * n_embd);
    }
    std::vector<float> dots;
    if (!embed_dots(m, backend, buft, ids, nk, gnext, N, dots, err)) return false;
    out.delta.assign((size_t) N * topk, NAN);
    out.spread.assign(N, NAN);
    for (int s = 0; s + 2 < N; ++s) { // s = N-2: no later prediction depends on x_{s+1}
        const float d_act = dots[(size_t) s * nk + topk];
        double sp = 0.0, sd = 0.0, sdd = 0.0;
        for (int r = 0; r < topk; ++r) {
            const double d = -(dots[(size_t) s * nk + r] - d_act);
            const double p = out.prob[(size_t) s * topk + r];
            out.delta[(size_t) s * topk + r] = (float) d;
            sp += p; sd += p * d; sdd += p * d * d;
        }
        if (sp > 0) {
            const double mean = sd / sp;
            out.spread[s] = (float) std::sqrt(std::max(0.0, sdd / sp - mean * mean));
        }
    }
    if (grad_emb) {
        grad_emb->resize(grad.size());
        for (size_t i = 0; i < grad.size(); ++i) (*grad_emb)[i] = -grad[i]; // log-likelihood sign
    }
    return true;
}

} // namespace

bool jlens_value_exact(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                       const std::vector<int32_t> & tokens, int slot_layer, int chunk, bool keep_vectors,
                       jlens_value_result & out, std::string & err, bool full_fwd) {
    if (slot_layer < 0 || slot_layer >= m.hparams.n_layer - 1) {
        err = "value: slot layer out of range";
        return false;
    }
    if (tokens.size() < 2) {
        err = "value: need at least 2 tokens";
        return false;
    }
    // the chunk graphs hold [n_tokens, chunk, n_head] attention scores per block; halve the chunk on OOM.
    // Chunks below 32 tokens lose precision (other CUDA matmul kernels; ~5% gradient error at 16).
    for (chunk = std::max(32, chunk); ; chunk /= 2) {
        bool oom = false;
        if (value_exact_once(m, backend, buft, tokens, slot_layer, chunk, keep_vectors, full_fwd, out, oom, err)) return true;
        if (!oom || chunk < 64) return false;
        fprintf(stderr, "jlens value: %s, retrying with chunk %d\n", err.c_str(), chunk / 2);
    }
}

bool jlens_branch_grad(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                       const std::vector<int32_t> & tokens, int topk, int chunk, int n_real, jlens_branch_result & out,
                       std::string & err, std::vector<float> * grad_emb, int max_group) {
    if (tokens.size() < 3 || topk < 1 || topk > 64) {
        err = "branch: need at least 3 tokens and 1 <= topk <= 64";
        return false;
    }
    for (chunk = std::max(32, chunk); ; chunk /= 2) {
        bool oom = false;
        if (branch_grad_once(m, backend, buft, tokens, topk, chunk, n_real, out, grad_emb, max_group, oom, err)) return true;
        if (!oom || chunk < 64) return false;
        fprintf(stderr, "jlens branch: %s, retrying with chunk %d\n", err.c_str(), chunk / 2);
    }
}

bool jlens_text_loglik(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                       const std::vector<int32_t> & tokens, int chunk, int n_real, const float * emb_add,
                       std::vector<double> & ll, std::string & err) {
    value_store st;
    if (!st.alloc(m, buft, (int) tokens.size(), { (int) m.hparams.n_layer - 1 }, (int) m.hparams.n_layer - 1, false)) {
        err = "value: store alloc failed (out of VRAM?)";
        return false;
    }
    bool oom = false;
    if (!forward_chunks(m, backend, buft, tokens, std::max(32, chunk), st, emb_add, oom, err)) return false;
    return head_pass(m, backend, buft, tokens, st, 1, n_real, ll, nullptr, nullptr, err);
}

bool jlens_embed_rows(const jlens_weights & m, ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                      const std::vector<int32_t> & ids, std::vector<float> & rows, std::string & err) {
    const int64_t n_embd = m.hparams.n_embd;
    const int n = (int) ids.size();
    ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * tid = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    ggml_set_input(tid);
    ggml_tensor * r = ggml_get_rows(c, m.get("token_embd.weight"), tid);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, r);
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    bool ok = ggml_gallocr_alloc_graph(galloc, gf);
    if (ok) {
        ggml_backend_tensor_set(tid, ids.data(), 0, n * sizeof(int32_t));
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        rows.resize((size_t) n * n_embd);
        ggml_backend_tensor_get(r, rows.data(), 0, rows.size() * sizeof(float));
    }
    ggml_gallocr_free(galloc);
    ggml_free(c);
    if (!ok) err = "value: embedding rows failed";
    return ok;
}
