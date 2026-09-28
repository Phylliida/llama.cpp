// jlens-bwtest: isolate backward-op discrepancies between CUDA and CPU.
// Builds a minimal scenario graph with grads, runs fwd+bwd on both backends
// with identical inputs, and compares the grad w.r.t. the activation input.
//
// usage: jlens-bwtest -m model.gguf --scenario mm|rms|rope|glu [--n 5]

#include "jlens.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void die(const char * msg) { fprintf(stderr, "jlens-bwtest: %s\n", msg); exit(1); }

struct run_result {
    std::vector<float> y;      // forward output
    std::vector<float> gx;     // grad w.r.t. x
    int64_t            ne_y;
};

static run_result run_backend(const std::string & backend_name, const std::string & model_path,
                              const std::string & scenario, int n_tokens,
                              const std::vector<float> & x_host, const std::vector<float> & v_host) {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    if (backend_name == "cuda") { backend = ggml_backend_cuda_init(0); buft = ggml_backend_cuda_buffer_type(0); }
    else                        { backend = ggml_backend_cpu_init();  buft = ggml_backend_cpu_buffer_type();  }
    if (!backend) die("backend init failed");

    // weights must live on the same backend as the compute
    jlens_model model;
    if (!jlens_model_load(model_path.c_str(), buft, model)) die("model load failed");
    const jlens_model & m = model;
    const int64_t d = m.hparams.n_embd;

    ggml_init_params iparams = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(iparams);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, /*grads=*/true);

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n_tokens);
    ggml_set_name(x, "x");
    ggml_set_param(x);

    ggml_tensor * y = nullptr;
    const char * wname = "blk.0.attn_q.weight";
    if (scenario == "mm") {
        y = ggml_mul_mat(ctx, m.get(wname), x);
    } else if (scenario == "rms") {
        ggml_tensor * n = ggml_rms_norm(ctx, x, m.hparams.norm_eps);
        n = ggml_mul(ctx, n, m.get("blk.0.attn_norm.weight"));
        y = ggml_mul_mat(ctx, m.get(wname), n);
    } else if (scenario == "rope") {
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_name(pos, "pos");
        ggml_tensor * q3 = ggml_reshape_3d(ctx, x, m.hparams.n_embd_head, m.hparams.n_head, n_tokens);
        y = ggml_rope_ext(ctx, q3, pos, nullptr, m.hparams.n_rot, 0, m.hparams.n_ctx_train,
                          m.hparams.rope_freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    } else if (scenario == "glu") {
        ggml_tensor * up_out = ggml_mul_mat(ctx, m.get("blk.0.ffn_up.weight"), x); // [2*n_ff, n]
        const int64_t n_ff = m.hparams.n_ff;
        ggml_tensor * gate = ggml_cont(ctx, ggml_view_2d(ctx, up_out, n_ff, n_tokens, up_out->nb[1], 0));
        ggml_tensor * up   = ggml_cont(ctx, ggml_view_2d(ctx, up_out, n_ff, n_tokens, up_out->nb[1],
                                                         (size_t) n_ff * up_out->nb[0]));
        y = ggml_swiglu_split(ctx, gate, up);
    } else {
        die("unknown scenario");
    }
    ggml_set_name(y, "y");
    ggml_build_forward_expand(gf, y);
    y->flags |= GGML_TENSOR_FLAG_LOSS;

    // caller-provided seed
    ggml_init_params sparams = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * sctx = ggml_init(sparams);
    ggml_tensor * x_buf   = ggml_dup_tensor(sctx, x);
    ggml_tensor * y_buf   = ggml_dup_tensor(sctx, y);
    ggml_tensor * pos_buf = scenario == "rope" ? ggml_new_tensor_1d(sctx, GGML_TYPE_I32, n_tokens) : nullptr;
    if (!ggml_backend_alloc_ctx_tensors_from_buft(sctx, buft)) die("alloc inputs failed");
    ggml_backend_tensor_set(x_buf, x_host.data(), 0, x_host.size() * sizeof(float));
    ggml_backend_tensor_set(y_buf, v_host.data(), 0, ggml_nelements(y) * sizeof(float));
    if (pos_buf) {
        std::vector<int32_t> posv(n_tokens);
        for (int i = 0; i < n_tokens; ++i) posv[i] = i;
        ggml_backend_tensor_set(pos_buf, posv.data(), 0, posv.size() * sizeof(int32_t));
    }

    // wire inputs: x and pos are leaves in gf; give them buffers via set from host after galloc?
    // simpler: make gf's x/pos reference our pre-allocated buffers
    x->data   = x_buf->data;
    x->buffer = x_buf->buffer;
    if (scenario == "rope") {
        // find pos leaf
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            if (strcmp(ggml_get_name(t), "pos") == 0) { t->data = pos_buf->data; t->buffer = pos_buf->buffer; }
        }
    }

    std::vector<ggml_tensor *> grad_accs(ggml_graph_n_nodes(gf), nullptr);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (ggml_graph_node(gf, i) == y) { grad_accs[i] = y_buf; break; }
    }
    ggml_cgraph * gb = ggml_graph_dup(ctx, gf, true);
    ggml_build_backward_expand(ctx, gb, grad_accs.data());
    ggml_tensor * gx = ggml_graph_get_grad(gb, x);
    if (!gx) die("no grad for x");
    ggml_set_output(gx);
    ggml_set_output(y);

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(galloc, gb)) die("galloc failed");
    if (ggml_backend_graph_compute(backend, gb) != GGML_STATUS_SUCCESS) die("compute failed");

    run_result r;
    r.ne_y = ggml_nelements(y);
    r.y.resize(r.ne_y);
    r.gx.resize(x_host.size());
    ggml_backend_tensor_get(y, r.y.data(), 0, r.y.size() * sizeof(float));
    ggml_backend_tensor_get(gx, r.gx.data(), 0, r.gx.size() * sizeof(float));
    ggml_backend_free(backend);
    return r;
}

static void compare(const char * what, const std::vector<float> & a, const std::vector<float> & b) {
    double max_abs = 0, max_ref = 0;
    for (size_t k = 0; k < a.size(); ++k) {
        max_abs = std::max(max_abs, (double) fabsf(a[k] - b[k]));
        max_ref = std::max(max_ref, (double) fabsf(a[k]));
    }
    printf("%s: max|d|=%.4e max|cuda|=%.4e rel=%.4e\n", what, max_abs, max_ref, max_abs / (max_ref + 1e-12));
}

int main(int argc, char ** argv) {
    std::string model_path, scenario = "mm";
    int n_tokens = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "-m")         model_path = argv[++i];
        else if (a == "--scenario") scenario   = argv[++i];
        else if (a == "--n")        n_tokens   = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty()) die("need -m");

    std::mt19937 rng(1234);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int64_t d = 6144; std::vector<float> x_host((size_t) d * n_tokens);
    for (auto & v : x_host) v = gauss(rng);
    // seed sized to scenario output; overprovision
    std::vector<float> v_host((size_t) d * 8 * n_tokens);
    for (auto & v : v_host) v = gauss(rng);

    run_result rc = run_backend("cuda", model_path, scenario, n_tokens, x_host, v_host);
    v_host.resize(rc.ne_y); // actual seed size used
    run_result rp = run_backend("cpu",  model_path, scenario, n_tokens, x_host, v_host);

    compare("forward y", rc.y, rp.y);
    compare("grad x   ", rc.gx, rp.gx);
    return 0;
}
