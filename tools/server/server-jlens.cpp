#include "server-jlens.h"

#include "jlens.h"

#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include "llama.h"
#include "src/llama-model.h"
#include "src/llama-vocab.h"

#include <algorithm>
#include <cfloat>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

// adapter exposing the server's already-loaded model weights through the
// jlens_weights interface used by the hand-built graphs
namespace {

struct server_jlens_weights : jlens_weights {
    std::unordered_map<std::string, ggml_tensor *> map;

    explicit server_jlens_weights(const llama_model * lm) {
        for (const auto & p : lm->tensors_by_name) {
            map[p.first] = p.second;
        }
        const auto & hp = lm->hparams;
        hparams.n_embd      = hp.n_embd;
        hparams.n_layer     = hp.n_layer();
        hparams.n_head      = hp.n_head();
        hparams.n_head_kv   = hp.n_head_kv();
        hparams.n_embd_head = hp.n_embd_head_k();
        hparams.n_ff        = hp.n_ff();
        hparams.n_rot       = hp.n_rot();
        hparams.n_ctx_train = hp.n_ctx_train;
        hparams.rope_freq_base = hp.rope_freq_base_train;
        hparams.norm_eps       = hp.f_norm_rms_eps;
        hparams.n_vocab        = get("output.weight")->ne[1];
    }

    ggml_tensor * get(const std::string & name) const override {
        auto it = map.find(name);
        return it == map.end() ? nullptr : it->second;
    }
};

struct server_jlens_state {
    std::mutex mutex;

    std::unique_ptr<server_jlens_weights> weights;

    // transport matrices in host RAM, already normalized (J = M / n_positions)
    int64_t n_embd  = 0;
    int64_t n_layer = 0;
    std::vector<std::vector<float>> J;

    ggml_backend_t backend = nullptr; // on the CUDA device holding the weights
    int            device  = -1;

    // head decode graph (fixed shape [n_embd] -> [n_vocab]), built once
    ggml_context * hctx  = nullptr;
    ggml_tensor *  x_in  = nullptr;
    ggml_cgraph *  hgf   = nullptr;
    ggml_tensor *  hlogits = nullptr;

    // dedicated allocator for the head graph so its transients can never
    // overlap forward-graph l_out storage
    ggml_gallocr_t hgalloc = nullptr;

    // staging buffer for one layer's J matrix on device (all-positions mode)
    ggml_context *         jctx = nullptr;
    ggml_backend_buffer_t  jbuf = nullptr;
    ggml_tensor *          jt   = nullptr;
    int                    jt_layer = -1;
    std::vector<float>     jt_host; // host scratch for the transpose before upload
};

server_jlens_state g_jlens;

// caller must hold g_jlens.mutex
void server_jlens_deinit_locked() {
    g_jlens.weights.reset(); // drop raw tensor pointers before the model dies
    g_jlens.J.clear();
    g_jlens.n_embd  = 0;
    g_jlens.n_layer = 0;
    if (g_jlens.hgalloc) { ggml_gallocr_free(g_jlens.hgalloc);      g_jlens.hgalloc = nullptr; }
    if (g_jlens.hctx)    { ggml_free(g_jlens.hctx);                 g_jlens.hctx    = nullptr; }
    if (g_jlens.jbuf)    { ggml_backend_buffer_free(g_jlens.jbuf);  g_jlens.jbuf    = nullptr; }
    if (g_jlens.jctx)    { ggml_free(g_jlens.jctx);                 g_jlens.jctx    = nullptr; }
    if (g_jlens.backend) { ggml_backend_free(g_jlens.backend);      g_jlens.backend = nullptr; }
    g_jlens.x_in     = nullptr;
    g_jlens.hgf      = nullptr;
    g_jlens.hlogits  = nullptr;
    g_jlens.jt       = nullptr;
    g_jlens.jt_layer = -1;
    g_jlens.jt_host.clear();
    g_jlens.device   = -1;
}

} // namespace

void server_jlens_deinit() {
    // blocks until any in-flight /jlens compute finishes
    std::lock_guard<std::mutex> lock(g_jlens.mutex);
    server_jlens_deinit_locked();
}

bool server_jlens_init(const llama_model * model, const std::string & jlens_path, std::string & err) {
    std::lock_guard<std::mutex> lock(g_jlens.mutex);
    server_jlens_deinit_locked(); // idempotent: safe re-init after model reload

    auto weights = std::make_unique<server_jlens_weights>(model);

    // every tensor the hand-built graphs need must exist and live on a single
    // CUDA device; compute runs there (buffer pointers are device-local)
    int dev = -1;
    auto check_tensor = [&](const std::string & name) -> bool {
        ggml_tensor * t = weights->get(name);
        if (!t) {
            err = "jlens: model is missing tensor " + name;
            return false;
        }
        if (!t->buffer) {
            err = "jlens: tensor has no backend buffer: " + name;
            return false;
        }
        const char * bn = ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer));
        if (strncmp(bn, "CUDA", 4) != 0) {
            err = "jlens: tensor is not on a CUDA device: " + name;
            return false;
        }
        const int d = atoi(bn + 4);
        if (dev < 0) {
            dev = d;
        } else if (dev != d) {
            err = "jlens: model weights span multiple CUDA devices";
            return false;
        }
        return true;
    };
    static const char * blk_names[] = {
        "blk.%d.attn_norm.weight",       "blk.%d.attn_q.weight",
        "blk.%d.attn_k.weight",          "blk.%d.attn_v.weight",
        "blk.%d.attn_output.weight",     "blk.%d.post_attention_norm.weight",
        "blk.%d.ffn_norm.weight",        "blk.%d.ffn_up.weight",
        "blk.%d.ffn_down.weight",        "blk.%d.post_ffw_norm.weight",
    };
    for (const char * n : {"token_embd.weight", "output_norm.weight", "output.weight"}) {
        if (!check_tensor(n)) {
            return false;
        }
    }
    for (int64_t il = 0; il < weights->hparams.n_layer; ++il) {
        for (const char * fmt : blk_names) {
            char buf[64];
            snprintf(buf, sizeof(buf), fmt, (int) il);
            if (!check_tensor(buf)) {
                return false;
            }
        }
    }

    // load + normalize the accumulators
    FILE * f = fopen(jlens_path.c_str(), "rb");
    if (!f) {
        err = "cannot open jlens file: " + jlens_path;
        return false;
    }
    uint32_t hdr[4];
    uint64_t n_probes = 0, n_positions = 0;
    if (fread(hdr, 4, 4, f) != 4 || hdr[0] != 0x4A4C4E53 || hdr[1] < 1 || hdr[1] > 2) {
        fclose(f);
        err = "bad jlens header: " + jlens_path;
        return false;
    }
    if (fread(&n_probes, 8, 1, f) != 1 || fread(&n_positions, 8, 1, f) != 1) {
        fclose(f);
        err = "bad jlens header: " + jlens_path;
        return false;
    }
    if (n_positions == 0) {
        fclose(f);
        err = "jlens file has 0 positions: " + jlens_path;
        return false;
    }
    const int64_t n_embd  = hdr[2];
    const int64_t n_layer = hdr[3];
    if (n_embd != weights->hparams.n_embd || n_layer != weights->hparams.n_layer) {
        fclose(f);
        err = "jlens/model shape mismatch";
        return false;
    }
    g_jlens.J.resize(n_layer);
    // J = M / n_positions: each probe contributes n_valid rank-1 samples (valid
    // source positions), so the positions counter already spans all probes
    const double scale = 1.0 / (double) n_positions;
    for (int64_t il = 0; il < n_layer; ++il) {
        g_jlens.J[il].resize((size_t) n_embd * n_embd);
        if (fread(g_jlens.J[il].data(), sizeof(float), g_jlens.J[il].size(), f) != g_jlens.J[il].size()) {
            fclose(f);
            g_jlens.J.clear();
            err = "short jlens file";
            return false;
        }
        for (auto & x : g_jlens.J[il]) {
            x = (float) (x * scale);
        }
        double norm2 = 0;
        for (const auto & x : g_jlens.J[il]) norm2 += (double) x * x;
        if (norm2 == 0) {
            fprintf(stderr, "srv jlens: WARNING: layer %lld is all zeros (not fitted; above the fitter's --target-layer?)\n",
                    (long long) il);
        }
    }
    fclose(f);
    g_jlens.n_embd  = n_embd;
    g_jlens.n_layer = n_layer;

    g_jlens.backend = ggml_backend_cuda_init(dev);
    if (!g_jlens.backend) {
        err = "jlens: cuda init failed";
        return false;
    }
    g_jlens.device = dev;
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(dev);

    // head decode graph, fixed shape
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    g_jlens.hctx = ggml_init(ip);
    g_jlens.hgf  = jlens_build_head(*weights, g_jlens.hctx, &g_jlens.x_in);
    if (!g_jlens.hgf) {
        err = "jlens: head graph build failed";
        return false;
    }
    g_jlens.hlogits = ggml_graph_get_tensor(g_jlens.hgf, "jlens_head_logits");
    g_jlens.hgalloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(g_jlens.hgalloc, g_jlens.hgf)) {
        err = "jlens: head graph alloc failed";
        return false;
    }

    // staging tensor for one layer's J (6144^2 f32 = 151 MB), uploaded on demand
    ggml_init_params jp = { 4 * ggml_tensor_overhead(), nullptr, true };
    g_jlens.jctx = ggml_init(jp);
    g_jlens.jt   = ggml_new_tensor_2d(g_jlens.jctx, GGML_TYPE_F32, n_embd, n_embd);
    g_jlens.jbuf = ggml_backend_alloc_ctx_tensors_from_buft(g_jlens.jctx, buft);
    if (!g_jlens.jbuf) {
        err = "jlens: J staging alloc failed";
        return false;
    }

    g_jlens.weights = std::move(weights);

    fprintf(stderr, "srv  jlens: loaded %" PRId64 " layers x [%" PRId64 " x %" PRId64 "], %" PRIu64 " positions (CUDA%d)\n",
            n_layer, n_embd, n_embd, n_positions, dev);
    return true;
}

bool server_jlens_enabled() {
    return g_jlens.weights != nullptr;
}

bool server_jlens_compute(
        const llama_vocab * vocab,
        const std::vector<int32_t> & tokens,
        int pos,
        int topk,
        const std::vector<int> * layers_in,
        bool all_positions,
        std::vector<server_jlens_layer_result> & out_layers,
        std::vector<std::pair<std::string, float>> & out_model_topk,
        std::string & err) {

    std::lock_guard<std::mutex> lock(g_jlens.mutex);
    if (!g_jlens.weights) {
        err = "jlens: not initialized (model reloading?)";
        return false;
    }

    const int n_tokens = (int) tokens.size();
    if (n_tokens == 0 || n_tokens > 8192) {
        err = "jlens: n_tokens out of range (1..8192)";
        return false;
    }
    if (pos < 0) {
        pos += n_tokens;
    }
    if (pos < 0 || pos >= n_tokens) {
        err = "jlens: pos out of range";
        return false;
    }
    if (topk < 1) {
        err = "jlens: topk must be >= 1";
        return false;
    }

    std::vector<int> layers;
    if (layers_in && !layers_in->empty()) {
        layers = *layers_in;
    } else {
        for (int il = 0; il < g_jlens.n_layer; il += 5) layers.push_back(il);
        if (layers.back() != (int) g_jlens.n_layer - 1) {
            layers.push_back((int) g_jlens.n_layer - 1);
        }
    }
    for (int il : layers) {
        if (il < 0 || il >= (int) g_jlens.n_layer) {
            err = "jlens: layer out of range [0, " + std::to_string(g_jlens.n_layer) + ")";
            return false;
        }
    }

    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(g_jlens.device);

    // forward pass exposing block outputs (flash attention, forward only).
    // with_head=false: skip the full-sequence logits GEMM + [n_vocab, n_tokens]
    // tensor; the model reference is decoded per-position through the head graph.
    jlens_forward fwd;
    if (!jlens_build_forward(*g_jlens.weights, n_tokens, /*with_grad_flags=*/false, fwd,
                             -1, /*use_flash_attn=*/true, /*with_head=*/false)) {
        err = "jlens: graph build failed";
        return false;
    }
    // pin only the activations we actually fetch: every ggml_set_output'd tensor
    // is retained by the allocator for the whole graph lifetime
    for (int il : layers) ggml_set_output(fwd.l_out[il]);
    ggml_set_output(fwd.l_out.back()); // model_topk reference decode

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(galloc, fwd.gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(fwd.gctx);
        err = "jlens: graph alloc failed (out of VRAM?)";
        return false;
    }

    {
        std::vector<int32_t> posv(n_tokens);
        for (int i = 0; i < n_tokens; ++i) posv[i] = i;
        std::vector<float> mask((size_t) n_tokens * n_tokens);
        for (int j = 0; j < n_tokens; ++j)
            for (int i = 0; i < n_tokens; ++i)
                mask[i + (size_t) j * n_tokens] = (i <= j) ? 0.0f : -FLT_MAX;
        ggml_backend_tensor_set(fwd.tokens, tokens.data(), 0, n_tokens * sizeof(int32_t));
        ggml_backend_tensor_set(fwd.pos,    posv.data(),   0, n_tokens * sizeof(int32_t));
        ggml_backend_tensor_set(fwd.mask,   mask.data(),   0, mask.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(g_jlens.backend, fwd.gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc);
        ggml_free(fwd.gctx);
        err = "jlens: forward compute failed";
        return false;
    }

    const int64_t n_embd  = g_jlens.n_embd;
    const int64_t n_vocab = g_jlens.weights->hparams.n_vocab;
    const int     n_real  = llama_vocab_n_tokens(vocab); // logits may be padded

    auto topk_of = [&](const std::vector<float> & logits) {
        std::vector<int> idx(logits.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
        const int n = std::min(topk, (int) idx.size());
        std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        std::vector<std::pair<std::string, float>> out;
        for (int i = 0; i < n && idx[i] < n_real; ++i) {
            out.emplace_back(llama_vocab_get_text(vocab, idx[i]), logits[idx[i]]);
        }
        return out;
    };

    // decode one [n_embd] activation through the head graph (rms norm + lm_head)
    auto head_decode = [&](const float * a, std::vector<float> & logits) -> bool {
        ggml_backend_tensor_set(g_jlens.x_in, a, 0, (size_t) n_embd * sizeof(float));
        if (ggml_backend_graph_compute(g_jlens.backend, g_jlens.hgf) != GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_backend_tensor_get(g_jlens.hlogits, logits.data(), 0, (size_t) n_vocab * sizeof(float));
        return true;
    };

    // full-model reference at pos: final block output through the head
    {
        std::vector<float> a(n_embd), full_logits(n_vocab);
        ggml_backend_tensor_get(fwd.l_out.back(), a.data(),
                                ((size_t) pos * n_embd) * sizeof(float), n_embd * sizeof(float));
        if (!head_decode(a.data(), full_logits)) {
            ggml_gallocr_free(galloc);
            ggml_free(fwd.gctx);
            err = "jlens: head compute failed";
            return false;
        }
        out_model_topk = topk_of(full_logits);
    }

    const int n_threads = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    out_layers.clear();

    if (all_positions) {
        // batched: for each layer, y = J_l . a_l for ALL positions on the GPU in
        // position chunks, decode through the head, top-k on CPU.
        ggml_tensor * out_norm = g_jlens.weights->get("output_norm.weight");
        ggml_tensor * out_w    = g_jlens.weights->get("output.weight");
        // separate allocator: chunk graphs must not reuse the buffer holding fwd.l_out
        ggml_gallocr_t galloc2 = ggml_gallocr_new(buft);
        const int chunk = 512;
        std::vector<float> lg;
        for (int il : layers) {
            if (g_jlens.jt_layer != il) {
                // mul_mat(A, x)[i] = sum_k A[k + i*n_embd] * x[k], but the JLNS
                // file stores J[i,j] at i + j*n_embd — so upload J transposed to
                // get y = J·a (matching the axpy form in the single-position path)
                const std::vector<float> & Jl = g_jlens.J[il];
                g_jlens.jt_host.resize(Jl.size());
                {
                    std::vector<std::thread> ts;
                    const int64_t rows_per = (n_embd + n_threads - 1) / n_threads;
                    for (int t = 0; t < n_threads; ++t) {
                        const int64_t r0 = t * rows_per, r1 = std::min(n_embd, r0 + rows_per);
                        if (r0 >= r1) break;
                        ts.emplace_back([&, r0, r1] {
                            for (int64_t i = r0; i < r1; ++i)
                                for (int64_t j = 0; j < n_embd; ++j)
                                    g_jlens.jt_host[(size_t) j * n_embd + i] = Jl[(size_t) i * n_embd + j];
                        });
                    }
                    for (auto & t : ts) t.join();
                }
                ggml_backend_tensor_set(g_jlens.jt, g_jlens.jt_host.data(), 0,
                                        g_jlens.jt_host.size() * sizeof(float));
                g_jlens.jt_layer = il;
            }
            server_jlens_layer_result lr;
            lr.layer = il;
            lr.positions.resize(n_tokens);
            for (int p0 = 0; p0 < n_tokens; p0 += chunk) {
                const int np = std::min(chunk, n_tokens - p0);
                ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
                ggml_context * c = ggml_init(ip);
                // leaf aliasing the forward activation slice (l_out is contiguous
                // [n_embd, n_tokens] f32, so a column range is contiguous too).
                // gallocr skips tensors with pre-set data (ggml-alloc.c:594), so
                // this reads the forward buffer in place; crucially, a plain
                // ggml_view_2d would make build_forward_expand pull the ENTIRE
                // forward graph (~2000 nodes) into this tiny chunk graph.
                ggml_tensor * av = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_embd, np);
                av->data   = (char *) fwd.l_out[il]->data + (size_t) p0 * fwd.l_out[il]->nb[1];
                av->buffer = fwd.l_out[il]->buffer;
                ggml_tensor * y  = ggml_mul_mat(c, g_jlens.jt, av);
                ggml_tensor * yn = ggml_rms_norm(c, y, g_jlens.weights->hparams.norm_eps);
                yn = ggml_mul(c, yn, out_norm);
                ggml_tensor * logits = ggml_mul_mat(c, out_w, yn);
                ggml_cgraph * gf = ggml_new_graph(c);
                ggml_build_forward_expand(gf, logits);
                if (!ggml_gallocr_alloc_graph(galloc2, gf)) {
                    ggml_free(c);
                    ggml_gallocr_free(galloc2);
                    ggml_gallocr_free(galloc);
                    ggml_free(fwd.gctx);
                    err = "jlens: chunk graph alloc failed";
                    return false;
                }
                if (ggml_backend_graph_compute(g_jlens.backend, gf) != GGML_STATUS_SUCCESS) {
                    ggml_free(c);
                    ggml_gallocr_free(galloc2);
                    ggml_gallocr_free(galloc);
                    ggml_free(fwd.gctx);
                    err = "jlens: chunk compute failed";
                    return false;
                }
                lg.resize((size_t) n_vocab * np);
                ggml_backend_tensor_get(logits, lg.data(), 0, lg.size() * sizeof(float));
                ggml_free(c);
                // per-position top-k, positions spread across threads
                {
                    std::vector<std::thread> ts;
                    const int per = (np + n_threads - 1) / n_threads;
                    for (int t = 0; t < n_threads; ++t) {
                        const int q0 = t * per, q1 = std::min(np, q0 + per);
                        if (q0 >= q1) break;
                        ts.emplace_back([&, q0, q1] {
                            std::vector<int> idx(n_vocab);
                            for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
                            for (int q = q0; q < q1; ++q) {
                                const float * col = lg.data() + (size_t) q * n_vocab;
                                const int k = std::min(topk, (int) n_vocab);
                                std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                                                  [&](int a2, int b2) { return col[a2] > col[b2]; });
                                auto & dst = lr.positions[p0 + q];
                                for (int i = 0; i < k && idx[i] < n_real; ++i) {
                                    dst.emplace_back(llama_vocab_get_text(vocab, idx[i]), col[idx[i]]);
                                }
                            }
                        });
                    }
                    for (auto & t : ts) t.join();
                }
            }
            out_layers.push_back(std::move(lr));
        }
        ggml_gallocr_free(galloc2);
    } else {
        // single position: y = J_l . a_l[:, pos] on CPU, decode through the head on GPU
        std::vector<float> a(n_embd), y(n_embd), lens_logits(n_vocab);
        for (int il : layers) {
            ggml_backend_tensor_get(fwd.l_out[il], a.data(),
                                    ((size_t) pos * n_embd) * sizeof(float), n_embd * sizeof(float));
            const std::vector<float> & Jl = g_jlens.J[il];
            // J is stored in ggml column-major layout: J[i,j] at offset
            // i + j*n_embd, so y = J·a is a column-wise axpy (row-wise dots
            // would compute Jᵀ·a)
            std::fill(y.begin(), y.end(), 0.0f);
            {
                std::vector<std::thread> ts;
                const int64_t rows_per = (n_embd + n_threads - 1) / n_threads;
                for (int t = 0; t < n_threads; ++t) {
                    const int64_t r0 = t * rows_per, r1 = std::min(n_embd, r0 + rows_per);
                    if (r0 >= r1) break;
                    ts.emplace_back([&, r0, r1] {
                        for (int64_t j = 0; j < n_embd; ++j) {
                            const float * col = Jl.data() + (size_t) j * n_embd;
                            const float aj = a[j];
                            for (int64_t i = r0; i < r1; ++i) y[i] += col[i] * aj;
                        }
                    });
                }
                for (auto & t : ts) t.join();
            }
            if (!head_decode(y.data(), lens_logits)) {
                ggml_gallocr_free(galloc);
                ggml_free(fwd.gctx);
                err = "jlens: head compute failed";
                return false;
            }

            server_jlens_layer_result lr;
            lr.layer = il;
            lr.topk  = topk_of(lens_logits);
            out_layers.push_back(std::move(lr));
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(fwd.gctx);
    return true;
}
