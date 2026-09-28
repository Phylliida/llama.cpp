// jlens-capture: forward pass over a GLM-4 dense GGUF with per-block residual
// capture + logit-lens smoke test (final norm + lm_head applied to any block's
// output at any position).
//
// usage:
//   jlens-capture -m model.gguf --tokens "151644,872,198,..." [--layer 60] [--pos -1] [--topk 10]

#include "jlens.h"

#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_topk(const std::vector<float> & logits, const std::vector<std::string> & vocab, int k, const char * tag) {
    std::vector<int> idx(logits.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    const int n = std::min<int>(k, idx.size());
    std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    printf("%s top-%d:", tag, n);
    for (int i = 0; i < n; ++i) {
        std::string tok = idx[i] < (int) vocab.size() ? vocab[idx[i]] : "?";
        // escape newlines for display
        std::string esc;
        for (char c : tok) { if (c == '\n') esc += "\\n"; else esc += c; }
        printf("  [%d] \"%s\" %.3f\n", idx[i], esc.c_str(), logits[idx[i]]);
    }
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string tokens_str;
    int layer = -1; // default: last
    int pos   = -1; // last position
    int topk  = 10;
    bool use_flash = false;
    std::string backend_name = "cuda";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m") model_path = argv[++i];
        else if (a == "--tokens") tokens_str = argv[++i];
        else if (a == "--layer") layer = atoi(argv[++i]);
        else if (a == "--pos")   pos   = atoi(argv[++i]);
        else if (a == "--topk")  topk  = atoi(argv[++i]);
        else if (a == "--flash") use_flash = true;
        else if (a == "--backend") backend_name = argv[++i];
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || tokens_str.empty()) {
        fprintf(stderr, "usage: jlens-capture -m model.gguf --tokens \"1,2,3\" [--layer N] [--pos P] [--topk K]\n");
        return 1;
    }

    std::vector<int32_t> ids;
    {
        char * s = strdup(tokens_str.c_str());
        for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) ids.push_back(atoi(t));
        free(s);
    }
    const int n_tokens = (int) ids.size();
    if (n_tokens < 2) { fprintf(stderr, "need at least 2 tokens\n"); return 1; }

    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    if (backend_name == "cuda") {
        backend = ggml_backend_cuda_init(0);
        buft = ggml_backend_cuda_buffer_type(0);
    } else if (backend_name == "cpu") {
        backend = ggml_backend_cpu_init();
        buft = ggml_backend_cpu_buffer_type();
    }
    if (!backend || !buft) { fprintf(stderr, "no %s backend\n", backend_name.c_str()); return 1; }

    jlens_model model;
    if (!jlens_model_load(model_path.c_str(), buft, model)) return 1;

    if (layer < 0) layer = (int) model.hparams.n_layer - 1;
    if (pos   < 0) pos   = n_tokens - 1;

    jlens_forward fwd;
    if (!jlens_build_forward(model, n_tokens, /*with_grad_flags=*/false, fwd, -1, use_flash)) return 1;

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(galloc, fwd.gf)) { fprintf(stderr, "galloc failed\n"); return 1; }

    // inputs
    {
        std::vector<int32_t> posv(n_tokens);
        for (int i = 0; i < n_tokens; ++i) posv[i] = i;
        std::vector<float> mask((size_t) n_tokens * n_tokens);
        for (int j = 0; j < n_tokens; ++j)
            for (int i = 0; i < n_tokens; ++i)
                mask[i + (size_t) j * n_tokens] = (i <= j) ? 0.0f : -FLT_MAX;

        ggml_backend_tensor_set(fwd.tokens, ids.data(),  0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(fwd.pos,    posv.data(), 0, posv.size() * sizeof(int32_t));
        ggml_backend_tensor_set(fwd.mask,   mask.data(), 0, mask.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(backend, fwd.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "forward compute failed\n"); return 1;
    }

    // full-model logits at pos
    const int64_t n_vocab = model.hparams.n_vocab;
    std::vector<float> full_logits(n_vocab);
    ggml_backend_tensor_get(fwd.logits, full_logits.data(),
                            (size_t) pos * n_vocab * sizeof(float), n_vocab * sizeof(float));
    print_topk(full_logits, model.vocab_tokens, topk, "full-model");

    // logit lens on requested block: copy activation column, run head graph
    const int64_t n_embd = model.hparams.n_embd;
    std::vector<float> act(n_embd);
    ggml_backend_tensor_get(fwd.l_out[layer], act.data(),
                            (size_t) pos * n_embd * sizeof(float), n_embd * sizeof(float));

    ggml_init_params hparams = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * hctx = ggml_init(hparams);
    ggml_tensor * x_in = nullptr;
    ggml_cgraph * hgf = jlens_build_head(model, hctx, &x_in);
    if (!ggml_gallocr_alloc_graph(galloc, hgf)) { fprintf(stderr, "galloc head failed\n"); return 1; }
    ggml_backend_tensor_set(x_in, act.data(), 0, act.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, hgf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "head compute failed\n"); return 1;
    }
    ggml_tensor * hlogits = ggml_graph_get_tensor(hgf, "jlens_head_logits");
    std::vector<float> lens_logits(n_vocab);
    ggml_backend_tensor_get(hlogits, lens_logits.data(), 0, n_vocab * sizeof(float));
    char tag[64];
    snprintf(tag, sizeof(tag), "logit-lens L%d", layer);
    print_topk(lens_logits, model.vocab_tokens, topk, tag);

    printf("captured %d block outputs, n_tokens=%d, pos=%d\n", (int) fwd.l_out.size(), n_tokens, pos);
    return 0;
}
