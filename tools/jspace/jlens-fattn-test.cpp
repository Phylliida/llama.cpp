// jlens-fattn-test: isolate flash-attention correctness for GLM-4 geometry
// (n_head=48, n_head_kv=2, head_dim=128) by comparing ggml_flash_attn_ext
// against an explicit mul_mat + softmax attention on identical random inputs.
//
// usage: jlens-fattn-test [n_tokens] [variant]
//   variant 0: q/k/v contiguous [n_embd_head, n_tokens, n_head(_kv)] (jspace style)
//   variant 1: q/k strided permutes, v transposed (llama style)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void die(const char * msg) { fprintf(stderr, "jlens-fattn-test: %s\n", msg); exit(1); }

int main(int argc, char ** argv) {
    const int n_tokens = argc > 1 ? atoi(argv[1]) : 5;
    const int variant  = argc > 2 ? atoi(argv[2]) : 0;

    const int64_t n_embd_head = 128;
    const int64_t n_head      = 48;
    const int64_t n_head_kv   = 2;
    const float   kq_scale    = 1.0f / sqrtf((float) n_embd_head);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) die("cuda init failed");
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(0);

    ggml_init_params ip = { 64 * ggml_tensor_overhead() + 2 * ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    // canonical q/k/v in [n_embd_head, n_head, n_tokens] layout (as produced by
    // reshape of the qkv projections)
    ggml_tensor * q0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head,    n_tokens);
    ggml_tensor * k0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, n_tokens);
    ggml_tensor * v0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, n_tokens);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_input(q0); ggml_set_input(k0); ggml_set_input(v0); ggml_set_input(mask);

    // explicit reference: scores + softmax + weighted values (heads-last)
    auto explicit_attn = [&](ggml_tensor * m, float scale) {
        ggml_tensor * qp_r = ggml_permute(ctx, q0, 0, 2, 1, 3);
        ggml_tensor * kp_r = ggml_cont(ctx, ggml_permute(ctx, k0, 0, 2, 1, 3));
        ggml_tensor * kq   = ggml_mul_mat(ctx, kp_r, qp_r); // [n_kv, n_q, n_head] via gqa broadcast
        kq = ggml_soft_max_ext(ctx, kq, m, scale, 0.0f);
        ggml_tensor * vp_r = ggml_cont(ctx, ggml_permute(ctx, v0, 1, 2, 0, 3)); // [n_tokens, n_embd_head, n_head]
        return ggml_mul_mat(ctx, vp_r, kq);                                     // [n_embd_head, n_q, n_head]
    };
    ggml_tensor * kqv_ref   = explicit_attn(mask, kq_scale); // correct
    ggml_tensor * kqv_nomask  = explicit_attn(nullptr, kq_scale);
    ggml_tensor * kqv_noscale = explicit_attn(mask, 1.0f);

    // flash variant
    ggml_tensor * qp = ggml_permute(ctx, q0, 0, 2, 1, 3);
    ggml_tensor * kp = ggml_permute(ctx, k0, 0, 2, 1, 3);
    ggml_tensor * vp = ggml_permute(ctx, v0, 0, 2, 1, 3);
    ggml_tensor * kf = nullptr;
    ggml_tensor * vf = nullptr;
    if (variant == 0) {
        qp = ggml_cont(ctx, qp);
        kp = ggml_cont(ctx, kp);
        vp = ggml_cont(ctx, vp);
        kf = ggml_cast(ctx, kp, GGML_TYPE_F16);
        vf = ggml_cast(ctx, vp, GGML_TYPE_F16);
    } else if (variant == 1) {
        // llama style: v stored transposed in cache -> pass v as [n_embd_head, n_kv] transposed
        vp = ggml_transpose(ctx, ggml_cont(ctx, vp));
        kf = ggml_cast(ctx, kp, GGML_TYPE_F16);
        vf = ggml_cast(ctx, vp, GGML_TYPE_F16);
    } else if (variant == 3) {
        // llama kv-cache strides: k view into padded [head_dim, n_kv_max, n_head_kv]
        // f16 cache; v transposed: [n_kv, head_dim, n_head_kv] with nb[0]=kv stride
        const int64_t n_kv_max = 256;
        ggml_tensor * kc = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_embd_head, n_kv_max, n_head_kv);
        ggml_tensor * vc = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_kv_max, n_embd_head, n_head_kv);
        ggml_set_input(kc); ggml_set_input(vc);
        ggml_set_name(kc, "kc"); ggml_set_name(vc, "vc");
        kf = ggml_view_3d(ctx, kc, n_embd_head, n_tokens, n_head_kv, kc->nb[1], kc->nb[2], 0);
        vf = ggml_view_3d(ctx, vc, n_tokens, n_embd_head, n_head_kv, vc->nb[1], vc->nb[2], 0);
        ggml_set_name(kf, "kf"); ggml_set_name(vf, "vf");
    } else {
        kf = ggml_cast(ctx, kp, GGML_TYPE_F16);
        vf = ggml_cast(ctx, vp, GGML_TYPE_F16);
    }
    ggml_tensor * mf = ggml_cast(ctx, mask, GGML_TYPE_F16);
    ggml_tensor * kqv_fa_raw = ggml_flash_attn_ext(ctx, qp, kf, vf, mf, kq_scale, 0.0f, 0.0f);
    ggml_prec_set_acc(kqv_fa_raw, GGML_PREC_F32);
    // flash_attn_ext output is ALREADY permuted back: [head_dim, n_head, n_q]
    // (ggml.c: ne = {v->ne[0], q->ne[2], q->ne[1], q->ne[3]}). Permute to the
    // reference layout [head_dim, n_q, n_head] for comparison.
    ggml_tensor * kqv_fa = ggml_cont(ctx, ggml_permute(ctx, kqv_fa_raw, 0, 2, 1, 3));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, kqv_ref);
    ggml_build_forward_expand(gf, kqv_nomask);
    ggml_build_forward_expand(gf, kqv_noscale);
    ggml_build_forward_expand(gf, kqv_fa);
    // roots with no consumers look dead to gallocr's lifetime analysis and get
    // aliased into reused buffer slots — pin everything we fetch afterwards
    ggml_set_output(kqv_ref);
    ggml_set_output(kqv_nomask);
    ggml_set_output(kqv_noscale);
    ggml_set_output(kqv_fa);

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) die("galloc failed");

    // random inputs, causal mask
    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    auto fill = [&](ggml_tensor * t) {
        std::vector<float> h(ggml_nelements(t));
        for (auto & x : h) x = gauss(rng);
        ggml_backend_tensor_set(t, h.data(), 0, h.size() * sizeof(float));
    };
    fill(q0); fill(k0); fill(v0);
    if (variant == 3) {
        // fill the f16 cache buffers with the f16-rounded k0/v0 values so the
        // f32 reference sees exactly what the kernel sees
        ggml_tensor * kc = ggml_graph_get_tensor(gf, "kc");
        ggml_tensor * vc = ggml_graph_get_tensor(gf, "vc");
        const int64_t n_kv_max = 256;
        std::vector<float> kh(ggml_nelements(k0)), vh(ggml_nelements(v0));
        ggml_backend_tensor_get(k0, kh.data(), 0, kh.size() * sizeof(float));
        ggml_backend_tensor_get(v0, vh.data(), 0, vh.size() * sizeof(float));
        // k0 is [head_dim, n_head_kv, n_tokens]; kc is [head_dim, n_kv_max, n_head_kv]
        std::vector<ggml_fp16_t> kc_h(n_embd_head * n_kv_max * n_head_kv, ggml_fp32_to_fp16(0.0f));
        std::vector<ggml_fp16_t> vc_h(n_kv_max * n_embd_head * n_head_kv, ggml_fp32_to_fp16(0.0f));
        for (int64_t h = 0; h < n_head_kv; ++h)
            for (int64_t t = 0; t < n_tokens; ++t)
                for (int64_t d = 0; d < n_embd_head; ++d) {
                    const float kv = kh[d + h * n_embd_head + t * n_embd_head * n_head_kv];
                    const float vv = vh[d + h * n_embd_head + t * n_embd_head * n_head_kv];
                    const ggml_fp16_t k16 = ggml_fp32_to_fp16(kv);
                    const ggml_fp16_t v16 = ggml_fp32_to_fp16(vv);
                    kc_h[d + t * n_embd_head + h * n_embd_head * n_kv_max] = k16;
                    vc_h[t + d * n_kv_max + h * n_kv_max * n_embd_head] = v16;
                    // round the reference inputs to f16 too
                    kh[d + h * n_embd_head + t * n_embd_head * n_head_kv] = ggml_fp16_to_fp32(k16);
                    vh[d + h * n_embd_head + t * n_embd_head * n_head_kv] = ggml_fp16_to_fp32(v16);
                }
        ggml_backend_tensor_set(kc, kc_h.data(), 0, kc_h.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(vc, vc_h.data(), 0, vc_h.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(k0, kh.data(), 0, kh.size() * sizeof(float));
        ggml_backend_tensor_set(v0, vh.data(), 0, vh.size() * sizeof(float));
    }
    {
        std::vector<float> m((size_t) n_tokens * n_tokens);
        for (int j = 0; j < n_tokens; ++j)
            for (int i = 0; i < n_tokens; ++i)
                m[i + (size_t) j * n_tokens] = (i <= j) ? 0.0f : -1e30f;
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) die("compute failed");

    const int64_t ne = n_embd_head * n_tokens * n_head;
    std::vector<float> a(ne), b(ne), c(ne), d(ne);
    ggml_backend_tensor_get(kqv_ref,     a.data(), 0, ne * sizeof(float));
    ggml_backend_tensor_get(kqv_fa,      b.data(), 0, ne * sizeof(float));
    ggml_backend_tensor_get(kqv_nomask,  c.data(), 0, ne * sizeof(float));
    ggml_backend_tensor_get(kqv_noscale, d.data(), 0, ne * sizeof(float));

    auto rel = [&](const std::vector<float> & x, const std::vector<float> & y) {
        double max_d = 0, max_x = 0;
        for (int64_t i = 0; i < ne; ++i) {
            max_d = std::max(max_d, (double) fabs(x[i] - y[i]));
            max_x = std::max(max_x, (double) fabs(x[i]));
        }
        return max_d / max_x;
    };
    printf("n_tokens=%d variant=%d: vs-ref=%.3e  vs-nomask=%.3e  vs-noscale=%.3e %s\n",
           n_tokens, variant, rel(a, b), rel(c, b), rel(d, b), rel(a, b) < 1e-3 ? "OK" : "MISMATCH");

    // per-head analysis: does flash head h match some reference head h'?
    // layout [n_embd_head, n_tokens, n_head], head slice = h * n_embd_head * n_tokens
    const int64_t hs = n_embd_head * n_tokens;
    for (int64_t h = 0; h < n_head; h += 7) {
        int64_t best = -1; double best_err = 1e30;
        for (int64_t h2 = 0; h2 < n_head; ++h2) {
            double err = 0, mag = 0;
            for (int64_t i = 0; i < hs; ++i) {
                err = std::max(err, (double) fabs(b[h * hs + i] - a[h2 * hs + i]));
                mag = std::max(mag, (double) fabs(a[h2 * hs + i]));
            }
            if (err / (mag + 1e-9) < best_err) { best_err = err / (mag + 1e-9); best = h2; }
        }
        printf("  flash head %2ld best-matches ref head %2ld (rel %.3e)\n", (long) h, (long) best, best_err);
    }
    return rel(a, b) < 1e-3 ? 0 : 1;
}
