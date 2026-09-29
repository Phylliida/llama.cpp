// jlens-readout: apply fitted J-lens transport matrices to a prompt's per-layer
// residual outputs and decode through the model's final norm + unembedding:
//
//   y_l = J_l . a_l[:, pos]        (J_l = M_l / n_positions; n_positions
//                                   accumulates n_tokens per probe, so it
//                                   already spans all probes)
//   logits_l = W_u . (rms_norm(y_l) * output_norm)
//
// usage: jlens-readout -m model.gguf --jlens accum.jlns --tokens "1,2,3"
//          [--pos -1] [--topk 10] [--layers "0,5,10,..."]

#include "jlens.h"

#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static void die(const char * msg) { fprintf(stderr, "jlens-readout: %s\n", msg); exit(1); }

static void print_topk(const std::vector<float> & logits, const std::vector<std::string> & vocab,
                       int k, const char * tag) {
    std::vector<int> idx(logits.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    const int n = std::min(k, (int) idx.size());
    std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    printf("%s top-%d:", tag, n);
    for (int i = 0; i < n; ++i) {
        std::string t = vocab[idx[i]];
        std::string esc;
        for (char c : t) { if (c == '\n') esc += "\\n"; else esc += c; }
        printf("  %s(%.2f)", esc.c_str(), logits[idx[i]]);
    }
    printf("\n");
}

int main(int argc, char ** argv) {
    std::string model_path, jlens_path, tokens_str, layers_str;
    int pos  = -1;
    int topk = 10;
    int n_threads = std::max(1u, std::thread::hardware_concurrency());

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "-m")       model_path = argv[++i];
        else if (a == "--jlens")  jlens_path = argv[++i];
        else if (a == "--tokens") tokens_str = argv[++i];
        else if (a == "--layers") layers_str = argv[++i];
        else if (a == "--pos")    pos        = atoi(argv[++i]);
        else if (a == "--topk")   topk       = atoi(argv[++i]);
        else if (a == "--threads") n_threads = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || jlens_path.empty() || tokens_str.empty()) {
        fprintf(stderr, "usage: jlens-readout -m model.gguf --jlens accum.jlns --tokens \"1,2,3\" "
                        "[--pos -1] [--topk K] [--layers \"0,5,...\"]\n");
        return 1;
    }

    std::vector<int32_t> ids;
    {
        char * s = strdup(tokens_str.c_str());
        for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) ids.push_back(atoi(t));
        free(s);
    }
    const int n_tokens = (int) ids.size();
    if (pos < 0) pos += n_tokens;
    if (pos < 0 || pos >= n_tokens) die("bad --pos");

    std::vector<int> layers;
    if (!layers_str.empty()) {
        char * s = strdup(layers_str.c_str());
        for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) layers.push_back(atoi(t));
        free(s);
    }

    // ---- load the JLNS accumulators (host RAM) ----
    FILE * f = fopen(jlens_path.c_str(), "rb");
    if (!f) die("cannot open jlens file");
    uint32_t hdr[4];
    if (fread(hdr, 4, 4, f) != 4 || hdr[0] != 0x4A4C4E53) die("bad jlens magic");
    const uint32_t version = hdr[1];
    const int64_t  n_embd  = hdr[2];
    const int64_t  n_layer = hdr[3];
    uint64_t n_probes, n_positions;
    if (fread(&n_probes, 8, 1, f) != 1 || fread(&n_positions, 8, 1, f) != 1) die("bad jlens header");
    if (version != 1) die("unsupported jlens version");
    fprintf(stderr, "jlens-readout: %lld layers x [%lld x %lld], %llu probes, %llu positions\n",
            (long long) n_layer, (long long) n_embd, (long long) n_embd,
            (unsigned long long) n_probes, (unsigned long long) n_positions);
    std::vector<std::vector<float>> J(n_layer, std::vector<float>((size_t) n_embd * n_embd));
    // J = M / n_positions: each probe chunk contributes n_tokens rank-1 samples, so the
    // positions counter already spans all probes (do NOT multiply by n_probes)
    if (n_positions == 0) die("jlens file has 0 positions");
    const double scale = 1.0 / (double) n_positions;
    for (int64_t il = 0; il < n_layer; ++il) {
        if (fread(J[il].data(), sizeof(float), J[il].size(), f) != J[il].size()) die("short jlens file");
        double norm2 = 0;
        for (auto & x : J[il]) { x = (float) (x * scale); norm2 += (double) x * x; }
        if (norm2 == 0)
            fprintf(stderr, "jlens-readout: WARNING: layer %lld is all zeros (not fitted; above the fitter's --target-layer?)\n",
                    (long long) il);
    }
    fclose(f);

    // ---- forward pass on the prompt ----
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) die("cuda init failed");
    jlens_model model;
    if (!jlens_model_load(model_path.c_str(), ggml_backend_cuda_buffer_type(0), model)) die("model load failed");
    if (model.hparams.n_embd != n_embd || model.hparams.n_layer != n_layer)
        die("jlens/model shape mismatch");
    const int64_t n_vocab = model.hparams.n_vocab;

    jlens_forward fwd;
    if (!jlens_build_forward(model, n_tokens, /*with_grad_flags=*/false, fwd)) die("graph build failed");
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cuda_buffer_type(0));
    for (auto * t : fwd.l_out) ggml_set_output(t);
    ggml_set_output(fwd.logits);
    if (!ggml_gallocr_alloc_graph(galloc, fwd.gf)) die("galloc failed");

    // inputs live in the compute buffer for the no-grads graph (plain leafs)
    {
        std::vector<int32_t> posv(n_tokens);
        for (int i = 0; i < n_tokens; ++i) posv[i] = i;
        std::vector<float> mask((size_t) n_tokens * n_tokens);
        for (int j = 0; j < n_tokens; ++j)
            for (int i = 0; i < n_tokens; ++i)
                mask[i + (size_t) j * n_tokens] = (i <= j) ? 0.0f : -FLT_MAX;
        ggml_backend_tensor_set(fwd.tokens, ids.data(),  0, n_tokens * sizeof(int32_t));
        ggml_backend_tensor_set(fwd.pos,    posv.data(), 0, posv.size() * sizeof(int32_t));
        ggml_backend_tensor_set(fwd.mask,   mask.data(), 0, mask.size() * sizeof(float));
    }
    if (ggml_backend_graph_compute(backend, fwd.gf) != GGML_STATUS_SUCCESS) die("compute failed");

    // full-model reference
    {
        std::vector<float> full_logits((size_t) n_vocab);
        ggml_backend_tensor_get(fwd.logits, full_logits.data(),
                                (size_t) pos * n_vocab * sizeof(float), n_vocab * sizeof(float));
        print_topk(full_logits, model.vocab_tokens, topk, "full-model");
    }

    // head graph for decoding transported activations; separate allocator so its
    // transients can never overlap the forward graph's pinned l_out storage
    ggml_init_params hparams = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * hctx = ggml_init(hparams);
    ggml_tensor *  x_in = nullptr;
    ggml_cgraph *  hgf  = jlens_build_head(model, hctx, &x_in);
    ggml_gallocr_t hgalloc = ggml_gallocr_new(ggml_backend_cuda_buffer_type(0));
    if (!ggml_gallocr_alloc_graph(hgalloc, hgf)) die("galloc head failed");
    ggml_tensor * hlogits = ggml_graph_get_tensor(hgf, "jlens_head_logits");
    std::vector<float> lens_logits(n_vocab);

    if (layers.empty()) {
        for (int il = 0; il < n_layer; il += 5) layers.push_back((int) il);
        if (layers.back() != n_layer - 1) layers.push_back((int) n_layer - 1);
    }

    // per-layer: y = J_l . a_l[:, pos], decode
    std::vector<float> a((size_t) n_embd), y((size_t) n_embd);
    for (int il : layers) {
        if (il < 0 || il >= n_layer) die("bad --layers entry");
        ggml_backend_tensor_get(fwd.l_out[il], a.data(),
                                ((size_t) pos * n_embd) * sizeof(float), n_embd * sizeof(float));
        const std::vector<float> & Jl = J[il];
        // J is stored in ggml column-major layout: J[i,j] at offset i + j*n_embd,
        // so y = J·a is a column-wise axpy accumulation (row-wise dot products
        // would compute Jᵀ·a instead)
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
        ggml_backend_tensor_set(x_in, y.data(), 0, y.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, hgf) != GGML_STATUS_SUCCESS) die("head compute failed");
        ggml_backend_tensor_get(hlogits, lens_logits.data(), 0, n_vocab * sizeof(float));
        char tag[64];
        snprintf(tag, sizeof(tag), "J-lens L%-2d", il);
        print_topk(lens_logits, model.vocab_tokens, topk, tag);
    }
    return 0;
}
