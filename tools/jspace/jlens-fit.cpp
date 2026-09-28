// jlens-fit: fit J-lens transport matrices for a GLM-4 dense GGUF.
//
// The transport matrix for block l is the corpus-averaged Jacobian
//   J_l = E_{prompt,position} [ d h_L,t / d a_l,t ]        (h_L = final block output)
// estimated with Gaussian random probes:
//   seed the backward pass at h_L with V ~ N(0, I) ([n_embd, n_tokens]),
//   read the VJP U_l = d(V . h_L)/d(a_l) at each block output,
//   accumulate  M_l += V U_l^T  ->  E[M_l] = E[J_l].
// Cross-position Jacobian terms (t' != t) vanish in expectation because the
// probe vectors are independent across positions, so all positions of a
// prompt are probed in a single backward pass.
//
// usage:
//   jlens-fit -m model.gguf --tokens "1,2,3,..." [options]
//     --backend cuda|cpu   compute backend (default cuda)
//     --n-probes N         number of probe backward passes (default 8)
//     --seed S             RNG seed for probes (default 42)
//     --out jlens.bin      write raw accumulator sums + counts (enables fitting)
//     --in  jlens.bin      load accumulators first (continue a run)
//     --dump-grads path    after probe 0, write all per-layer VJPs U_l and exit
//     --compare path       after probe 0, compare U_l against a --dump-grads file
//
// file format (little endian): "JLNS" u32, version u32, n_embd u32, n_layer u32,
//   n_probes u64, n_positions u64, then n_layer * n_embd * n_embd f32 row-major.
//   J_l = M_l / (n_probes * n_positions) is left to the readout side.

#include "jlens.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static void die(const char * msg) {
    fprintf(stderr, "jlens-fit: %s\n", msg);
    exit(1);
}

// ---- accumulator file ----

static bool acc_save(const char * path, const std::vector<std::vector<float>> & M,
                     int64_t n_embd, uint64_t n_probes, uint64_t n_positions) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;
    const uint32_t magic = 0x4A4C4E53; // "JLNS"
    const uint32_t version = 1;
    const uint32_t ne = (uint32_t) n_embd;
    const uint32_t nl = (uint32_t) M.size();
    fwrite(&magic,   4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&ne,      4, 1, f);
    fwrite(&nl,      4, 1, f);
    fwrite(&n_probes,    8, 1, f);
    fwrite(&n_positions, 8, 1, f);
    for (const auto & m : M) fwrite(m.data(), sizeof(float), m.size(), f);
    fclose(f);
    return true;
}

static bool acc_load(const char * path, std::vector<std::vector<float>> & M,
                     int64_t n_embd, uint64_t & n_probes, uint64_t & n_positions) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    uint32_t magic, version, ne, nl;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x4A4C4E53) { fclose(f); return false; }
    if (fread(&version, 4, 1, f) != 1 || version != 1)       { fclose(f); return false; }
    if (fread(&ne, 4, 1, f) != 1 || ne != (uint32_t) n_embd) { fclose(f); return false; }
    if (fread(&nl, 4, 1, f) != 1)                            { fclose(f); return false; }
    if (fread(&n_probes, 8, 1, f) != 1 || fread(&n_positions, 8, 1, f) != 1) { fclose(f); return false; }
    M.resize(nl);
    for (auto & m : M) {
        m.resize((size_t) n_embd * n_embd);
        if (fread(m.data(), sizeof(float), m.size(), f) != m.size()) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

// M[row-major d x d] += P, multithreaded over rows
static void acc_add(std::vector<float> & M, const float * P, int64_t d, int n_threads) {
    const int64_t rows_per = (d + n_threads - 1) / n_threads;
    std::vector<std::thread> ts;
    for (int t = 0; t < n_threads; ++t) {
        const int64_t r0 = t * rows_per;
        const int64_t r1 = std::min<int64_t>(d, r0 + rows_per);
        if (r0 >= r1) break;
        ts.emplace_back([&, r0, r1] {
            float * __restrict m = M.data() + r0 * d;
            const float * __restrict p = P + r0 * d;
            const int64_t n = (r1 - r0) * d;
            for (int64_t i = 0; i < n; ++i) m[i] += p[i];
        });
    }
    for (auto & t : ts) t.join();
}

int main(int argc, char ** argv) {
    std::string model_path, tokens_str, out_path, in_path, dump_path, compare_path, tokens_file;
    std::string backend_name = "cuda";
    int n_probes = 8;
    int seed     = 42;
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    int fd_layer = -1, fd_block = -1, fd_k = 6;
    float fd_eps = 0.5f;
    int chunk_len = 128, n_chunks = 0, stride = 0; // stride 0 -> chunk_len (non-overlapping)
    int save_every = 50; // checkpoint accumulators every N chunks (0 = only at end)
    bool lrp = false; // --lrp: fit the R-lens (LRP backward rules) instead of the J-lens

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "-m")         model_path   = argv[++i];
        else if (a == "--tokens")   tokens_str   = argv[++i];
        else if (a == "--tokens-file") tokens_file = argv[++i];
        else if (a == "--chunk-len")   chunk_len = atoi(argv[++i]);
        else if (a == "--n-chunks")    n_chunks  = atoi(argv[++i]);
        else if (a == "--stride")      stride    = atoi(argv[++i]);
        else if (a == "--save-every")  save_every = atoi(argv[++i]);
        else if (a == "--backend")  backend_name = argv[++i];
        else if (a == "--n-probes") n_probes     = atoi(argv[++i]);
        else if (a == "--seed")     seed         = atoi(argv[++i]);
        else if (a == "--threads")  n_threads    = atoi(argv[++i]);
        else if (a == "--out")      out_path     = argv[++i];
        else if (a == "--in")       in_path      = argv[++i];
        else if (a == "--dump-grads") dump_path   = argv[++i];
        else if (a == "--compare")    compare_path = argv[++i];
        else if (a == "--fd-check")   fd_layer = atoi(argv[++i]);
        else if (a == "--fd-block")   fd_block = atoi(argv[++i]);
        else if (a == "--fd-k")       fd_k     = atoi(argv[++i]);
        else if (a == "--fd-eps")     fd_eps   = (float) atof(argv[++i]);
        else if (a == "--lrp")        lrp        = true;
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || (tokens_str.empty() && tokens_file.empty())) {
        fprintf(stderr, "usage: jlens-fit -m model.gguf (--tokens \"1,2,3\" | --tokens-file ids.bin "
                        "[--chunk-len N] [--n-chunks N] [--stride N]) [--backend cuda|cpu] "
                        "[--n-probes N] [--out jlens.bin] [--dump-grads f | --compare f]\n");
        return 1;
    }

    std::vector<int32_t> ids;
    int n_tokens = 0;
    if (!tokens_file.empty()) {
        // raw little-endian i32 token id stream (see jlens-prep)
        FILE * f = fopen(tokens_file.c_str(), "rb");
        if (!f) die("cannot open tokens file");
        fseek(f, 0, SEEK_END);
        const long nbytes = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (nbytes <= 0 || nbytes % 4 != 0) { fclose(f); die("tokens file size not a multiple of 4"); }
        ids.resize(nbytes / 4);
        if (fread(ids.data(), 4, ids.size(), f) != ids.size()) { fclose(f); die("short tokens file"); }
        fclose(f);
        n_tokens = chunk_len;
        if (stride == 0) stride = chunk_len;
        if ((int) ids.size() < n_tokens) die("tokens file shorter than one chunk");
        const int avail = (int) ((ids.size() - n_tokens) / stride) + 1;
        if (n_chunks == 0 || n_chunks > avail) n_chunks = avail;
        fprintf(stderr, "jlens-fit: %zu tokens from %s, chunk_len=%d stride=%d -> %d chunks\n",
                ids.size(), tokens_file.c_str(), n_tokens, stride, n_chunks);
    } else {
        char * s = strdup(tokens_str.c_str());
        for (char * t = strtok(s, ","); t; t = strtok(nullptr, ",")) ids.push_back(atoi(t));
        free(s);
        n_tokens = (int) ids.size();
        n_chunks = 1;
        stride   = 0;
    }
    if (n_tokens < 2) die("need at least 2 tokens");
    if (lrp && fd_layer >= 0) die("--lrp has no perturb input; use --fd-block or drop --fd-check");

    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    if (backend_name == "cuda") {
        backend = ggml_backend_cuda_init(0);
        buft = ggml_backend_cuda_buffer_type(0);
    } else if (backend_name == "cpu") {
        backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, n_threads);
        buft = ggml_backend_cpu_buffer_type();
    } else {
        die("--backend must be cuda or cpu");
    }
    if (!backend) die("backend init failed");

    jlens_model model;
    if (!jlens_model_load(model_path.c_str(), buft, model)) die("model load failed");
    const int64_t n_embd  = model.hparams.n_embd;
    const int64_t n_layer = model.hparams.n_layer;

    jlens_forward fwd_plain;
    jlens_lrp_forward fwd_lrp;
    jlens_forward & fwd = lrp ? static_cast<jlens_forward &>(fwd_lrp) : fwd_plain;
    if (lrp) {
        if (!jlens_build_forward_lrp(model, n_tokens, fwd_lrp)) die("lrp graph build failed");
    } else {
        if (!jlens_build_forward(model, n_tokens, /*with_grad_flags=*/true, fwd_plain, fd_layer)) die("graph build failed");
    }
    if (!fwd.zero_pad) die("graph missing zero_pad param leaf");
    if (fd_layer >= 0 && !fwd.perturb) die("graph missing perturb input");

    // the VJP seed tensor: caller-provided grad accumulator for the loss node
    // (final block output), filled with a fresh Gaussian probe per pass
    ggml_init_params sparams = { 4 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
    ggml_context * sctx = ggml_init(sparams);
    ggml_tensor * vjp_seed = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(vjp_seed, "jlens_vjp_seed");
    if (!ggml_backend_alloc_ctx_tensors_from_buft(sctx, buft)) die("seed buffer alloc failed");

    // locate the loss node (final block output) in the forward graph
    std::vector<ggml_tensor *> grad_accs(ggml_graph_n_nodes(fwd.gf), nullptr);
    bool found_loss = false;
    for (int i = 0; i < ggml_graph_n_nodes(fwd.gf); ++i) {
        if (ggml_graph_node(fwd.gf, i) == fwd.l_out.back()) {
            grad_accs[i] = vjp_seed;
            found_loss = true;
            break;
        }
    }
    if (!found_loss) die("loss node not found in forward graph");

    ggml_cgraph * gb = ggml_graph_dup(fwd.gctx, fwd.gf, /*force_grads=*/true);
    ggml_build_backward_expand(fwd.gctx, gb, grad_accs.data());
    if (getenv("JLENS_DOT")) ggml_graph_dump_dot(gb, nullptr, getenv("JLENS_DOT"));
    fprintf(stderr, "jlens-fit: graph nodes fwd=%d fwd+bwd=%d\n", ggml_graph_n_nodes(fwd.gf), ggml_graph_n_nodes(gb));

    // per-layer VJP result tensors (transient grads, recomputed every pass)
    std::vector<ggml_tensor *> u_t(n_layer, nullptr);
    for (int il = 0; il < n_layer; ++il) {
        u_t[il] = ggml_graph_get_grad(gb, fwd.l_out[il]);
        if (!u_t[il]) die("no grad tensor for a block output");
        // critical: nothing in the graph consumes these grads, so ggml_gallocr would
        // consider their lifetimes over and alias them all into one buffer slot
        ggml_set_output(u_t[il]);
    }
    if (fd_layer >= 0) ggml_set_output(fwd.l_out.back()); // fetched in fd mode

    // Give ALL graph inputs dedicated storage outside ggml_gallocr. We observed
    // gallocr placing a transient tensor exactly on the gallocr-allocated tokens
    // leaf (offset 0), corrupting token ids mid-compute — params not being
    // registered as leafs (ggml.c:7281) was one instance; rather than chase every
    // lifetime-analysis edge case in this ggml version, bypass it for inputs.
    ggml_init_params inparams = { (8 + 7 * n_layer) * ggml_tensor_overhead(), nullptr, true };
    ggml_context * inctx = ggml_init(inparams);
    ggml_tensor * t_tokens  = ggml_dup_tensor(inctx, fwd.tokens);
    ggml_tensor * t_pos     = ggml_dup_tensor(inctx, fwd.pos);
    ggml_tensor * t_mask    = ggml_dup_tensor(inctx, fwd.mask);
    ggml_tensor * t_zero    = ggml_dup_tensor(inctx, fwd.zero_pad);
    ggml_tensor * t_perturb = fwd.perturb ? ggml_dup_tensor(inctx, fwd.perturb) : nullptr;
    std::vector<ggml_tensor *> t_leaves; // lrp detach-leaf storage
    if (lrp) {
        for (int il = 0; il < n_layer; ++il) {
            const jlens_lrp_leaves & lv = fwd_lrp.leaves[il];
            for (int k = 0; k < 4; ++k) t_leaves.push_back(ggml_dup_tensor(inctx, lv.rms_inv[k]));
            t_leaves.push_back(ggml_dup_tensor(inctx, lv.sig_gate));
            t_leaves.push_back(ggml_dup_tensor(inctx, lv.silu_gate));
            t_leaves.push_back(ggml_dup_tensor(inctx, lv.up));
        }
    }
    if (!ggml_backend_alloc_ctx_tensors_from_buft(inctx, buft)) die("input buffer alloc failed");
    auto wire = [](ggml_tensor * dst, const ggml_tensor * src) {
        dst->data   = src->data;
        dst->buffer = src->buffer;
    };
    wire(fwd.tokens, t_tokens);
    wire(fwd.pos, t_pos);
    wire(fwd.mask, t_mask);
    wire(fwd.zero_pad, t_zero);
    if (fwd.perturb) wire(fwd.perturb, t_perturb);
    if (lrp) {
        int li = 0;
        for (int il = 0; il < n_layer; ++il) {
            const jlens_lrp_leaves & lv = fwd_lrp.leaves[il];
            for (int k = 0; k < 4; ++k) wire(lv.rms_inv[k], t_leaves[li++]);
            wire(lv.sig_gate,  t_leaves[li++]);
            wire(lv.silu_gate, t_leaves[li++]);
            wire(lv.up,        t_leaves[li++]);
        }
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(galloc, gb)) die("galloc failed");
    fprintf(stderr, "jlens-fit: compute buffer %.2f GiB\n",
            ggml_gallocr_get_buffer_size(galloc, 0) / (1024.0 * 1024.0 * 1024.0));

    // static inputs
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
        const float zero = 0.0f;
        ggml_backend_tensor_set(fwd.zero_pad, &zero, 0, sizeof(float));
    }

    // ---- R-lens phase 1: detach-value harvest (per chunk, amortized over probes) ----
    jlens_forward dt;
    ggml_gallocr_t dt_galloc = nullptr;
    std::vector<ggml_tensor *> dt_rms;  // [n_layer * 4] norm inputs
    std::vector<ggml_tensor *> dt_gate; // [n_layer]
    std::vector<ggml_tensor *> dt_up;   // [n_layer]
    std::vector<float> h_rms((size_t) n_embd * n_tokens);
    std::vector<float> h_gate, h_up, h_sig, h_silu;
    std::vector<float> h_rms_inv;
    if (lrp) {
        if (!jlens_build_forward_dt(model, n_tokens, dt)) die("dt graph build failed");
        // shares token/pos/mask storage with the main graph
        wire(dt.tokens, t_tokens);
        wire(dt.pos,    t_pos);
        wire(dt.mask,   t_mask);
        dt_galloc = ggml_gallocr_new(buft);
        if (!ggml_gallocr_alloc_graph(dt_galloc, dt.gf)) die("dt galloc failed");
        fprintf(stderr, "jlens-fit: dt compute buffer %.2f GiB\n",
                ggml_gallocr_get_buffer_size(dt_galloc, 0) / (1024.0 * 1024.0 * 1024.0));
        for (int il = 0; il < n_layer; ++il) {
            char name[64];
            for (int k = 0; k < 4; ++k) {
                snprintf(name, sizeof(name), "jlens_dt_%d_rms_%d", il, k);
                ggml_tensor * t = ggml_graph_get_tensor(dt.gf, name);
                if (!t) die("dt rms source missing");
                dt_rms.push_back(t);
            }
            snprintf(name, sizeof(name), "jlens_dt_%d_gate", il);
            ggml_tensor * g = ggml_graph_get_tensor(dt.gf, name);
            snprintf(name, sizeof(name), "jlens_dt_%d_up", il);
            ggml_tensor * u = ggml_graph_get_tensor(dt.gf, name);
            if (!g || !u) die("dt gate/up source missing");
            dt_gate.push_back(g);
            dt_up.push_back(u);
        }
        const int64_t n_ff = model.hparams.n_ff;
        h_gate.resize((size_t) n_ff * n_tokens);
        h_up.resize((size_t) n_ff * n_tokens);
        h_sig.resize((size_t) n_ff * n_tokens);
        h_silu.resize((size_t) n_ff * n_tokens);
        h_rms_inv.resize(n_tokens);
    }
    auto harvest = [&]() {
        const int64_t n_ff = model.hparams.n_ff;
        const float eps = model.hparams.norm_eps;
        if (ggml_backend_graph_compute(backend, dt.gf) != GGML_STATUS_SUCCESS) die("dt compute failed");
        for (int il = 0; il < n_layer; ++il) {
            const jlens_lrp_leaves & lv = fwd_lrp.leaves[il];
            for (int k = 0; k < 4; ++k) {
                ggml_backend_tensor_get(dt_rms[il * 4 + k], h_rms.data(), 0, h_rms.size() * sizeof(float));
                // rms_inv[j] = 1/sqrt(mean_i x[i,j]^2 + eps); layout [n_embd, n_tokens]
                for (int j = 0; j < n_tokens; ++j) {
                    const float * x = h_rms.data() + (size_t) j * n_embd;
                    double ss = 0;
                    for (int64_t i = 0; i < n_embd; ++i) ss += (double) x[i] * x[i];
                    h_rms_inv[j] = (float) (1.0 / sqrt(ss / n_embd + eps));
                }
                ggml_backend_tensor_set(lv.rms_inv[k], h_rms_inv.data(), 0, n_tokens * sizeof(float));
            }
            ggml_backend_tensor_get(dt_gate[il], h_gate.data(), 0, h_gate.size() * sizeof(float));
            ggml_backend_tensor_get(dt_up[il],   h_up.data(),   0, h_up.size()   * sizeof(float));
            for (size_t i = 0; i < h_gate.size(); ++i) {
                const float sg = 1.0f / (1.0f + expf(-h_gate[i]));
                h_sig[i]  = sg;
                h_silu[i] = h_gate[i] * sg;
            }
            ggml_backend_tensor_set(lv.sig_gate,  h_sig.data(),  0, h_sig.size()  * sizeof(float));
            ggml_backend_tensor_set(lv.silu_gate, h_silu.data(), 0, h_silu.size() * sizeof(float));
            ggml_backend_tensor_set(lv.up,        h_up.data(),   0, h_up.size()   * sizeof(float));
        }
    };

    // outer-product graph for the rank-n_tokens accumulator updates:
    //   P = out_prod(V, U)  [n_embd, n_embd],  P[i,j] = sum_k V[i,k]*U[j,k] = (V U^T)[i,j]
    // (ggml_out_prod contracts over ne[1] of both operands)
    //
    // On CUDA the operands are wired in place: av reads the seed buffer directly and
    // au is rewired per layer onto the grad tensor's memory inside the main compute
    // buffer, so the GPU does the 4.8-GFLOP product and only its 151-MB result
    // crosses PCIe once per layer. On CPU we memcpy V/U into av/au as before.
    const bool acc_on_gpu = (backend_name == "cuda");
    ggml_backend_t acc_backend = acc_on_gpu ? backend : ggml_backend_cpu_init();
    if (!acc_on_gpu) ggml_backend_cpu_set_n_threads(acc_backend, n_threads);
    ggml_init_params aparams = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * actx = ggml_init(aparams);
    ggml_tensor * av = ggml_new_tensor_2d(actx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * au = ggml_new_tensor_2d(actx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * aP = ggml_out_prod(actx, av, au);
    ggml_cgraph * acc_gf = ggml_new_graph_custom(actx, 8, false);
    ggml_build_forward_expand(acc_gf, aP);
    if (acc_on_gpu) {
        av->data   = vjp_seed->data;
        av->buffer = vjp_seed->buffer;
    }
    ggml_gallocr_t acc_galloc = ggml_gallocr_new(acc_on_gpu ? buft : ggml_backend_cpu_buffer_type());
    if (!ggml_gallocr_alloc_graph(acc_galloc, acc_gf)) die("acc galloc failed");

    // host buffers
    std::vector<float> v_host((size_t) n_embd * n_tokens);
    std::vector<float> u_host((size_t) n_embd * n_tokens);
    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    // single-block fd validation: perturb the block input directly (short span ->
    // fp32 noise floor far below signal, unlike the full-model check above)
    if (fd_block >= 1) {
        // populate full-model activations, fetch the block input
        for (auto & x : v_host) x = gauss(rng);
        ggml_backend_tensor_set(vjp_seed, v_host.data(), 0, v_host.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, gb) != GGML_STATUS_SUCCESS) die("compute failed");
        std::vector<float> bin(u_host.size());
        ggml_backend_tensor_get(fwd.l_out[fd_block - 1], bin.data(), 0, bin.size() * sizeof(float));

        jlens_block blk;
        if (!jlens_build_block(model, fd_block, n_tokens, blk)) die("block graph build failed");

        ggml_init_params s2params = { 4 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
        ggml_context * s2ctx = ggml_init(s2params);
        ggml_tensor * seed2 = ggml_new_tensor_2d(s2ctx, GGML_TYPE_F32, n_embd, n_tokens);
        if (!ggml_backend_alloc_ctx_tensors_from_buft(s2ctx, buft)) die("seed2 alloc failed");

        std::vector<ggml_tensor *> ga2(ggml_graph_n_nodes(blk.gf), nullptr);
        bool found2 = false;
        for (int i = 0; i < ggml_graph_n_nodes(blk.gf); ++i) {
            if (ggml_graph_node(blk.gf, i) == blk.h_out) { ga2[i] = seed2; found2 = true; break; }
        }
        if (!found2) die("block loss node not found");

        ggml_cgraph * gb2 = ggml_graph_dup(blk.gctx, blk.gf, /*force_grads=*/true);
        ggml_build_backward_expand(blk.gctx, gb2, ga2.data());
        ggml_tensor * ux = ggml_graph_get_grad(gb2, blk.x);
        if (!ux) die("no grad for block input");
        ggml_set_output(ux);
        ggml_set_output(blk.h_out);

        ggml_gallocr_t galloc2 = ggml_gallocr_new(buft);
        if (!ggml_gallocr_alloc_graph(galloc2, gb2)) die("galloc2 failed");

        // dedicated storage for block inputs (same gallocr-aliasing workaround)
        ggml_init_params i2params = { 8 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * i2ctx = ggml_init(i2params);
        ggml_tensor * t2_h = ggml_dup_tensor(i2ctx, blk.h_in);
        ggml_tensor * t2_p = ggml_dup_tensor(i2ctx, blk.pos);
        ggml_tensor * t2_m = ggml_dup_tensor(i2ctx, blk.mask);
        ggml_tensor * t2_z = ggml_dup_tensor(i2ctx, blk.zero_pad);
        if (!ggml_backend_alloc_ctx_tensors_from_buft(i2ctx, buft)) die("block input alloc failed");
        auto wire2 = [](ggml_tensor * dst, const ggml_tensor * src) {
            dst->data = src->data; dst->buffer = src->buffer;
        };
        wire2(blk.h_in, t2_h); wire2(blk.pos, t2_p); wire2(blk.mask, t2_m); wire2(blk.zero_pad, t2_z);

        {   // static inputs
            std::vector<int32_t> posv(n_tokens);
            for (int i = 0; i < n_tokens; ++i) posv[i] = i;
            std::vector<float> mask((size_t) n_tokens * n_tokens);
            for (int j = 0; j < n_tokens; ++j)
                for (int i = 0; i < n_tokens; ++i)
                    mask[i + (size_t) j * n_tokens] = (i <= j) ? 0.0f : -FLT_MAX;
            ggml_backend_tensor_set(blk.pos, posv.data(), 0, posv.size() * sizeof(int32_t));
            ggml_backend_tensor_set(blk.mask, mask.data(), 0, mask.size() * sizeof(float));
            const float zero = 0.0f;
            ggml_backend_tensor_set(blk.zero_pad, &zero, 0, sizeof(float));
            ggml_backend_tensor_set(seed2, v_host.data(), 0, v_host.size() * sizeof(float));
        }

        // reference VJP at the unperturbed input
        ggml_backend_tensor_set(blk.h_in, bin.data(), 0, bin.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, gb2) != GGML_STATUS_SUCCESS) die("block compute failed");
        // whiten the probe: divide V elementwise by |h_out| (floored at 1) so massive-
        // activation dims stop dominating the fd dot's fp32 rounding noise. U = J^T V
        // stays exact for whatever V we choose, so recompute the reference after.
        {
            std::vector<float> h0(u_host.size());
            ggml_backend_tensor_get(blk.h_out, h0.data(), 0, h0.size() * sizeof(float));
            for (size_t i = 0; i < v_host.size(); ++i)
                v_host[i] /= (fabsf(h0[i]) > 1.0f ? fabsf(h0[i]) : 1.0f);
            ggml_backend_tensor_set(seed2, v_host.data(), 0, v_host.size() * sizeof(float));
            if (ggml_backend_graph_compute(backend, gb2) != GGML_STATUS_SUCCESS) die("block compute failed");
        }
        ggml_backend_tensor_get(ux, u_host.data(), 0, u_host.size() * sizeof(float));

        std::vector<float> h_host(u_host.size());
        std::vector<float> pbuf(bin.size());
        std::vector<float> dir(bin.size());
        std::mt19937 frng(7);
        printf("fd-block %d: %d random directions (block-input VJP vs central difference)\n", fd_block, fd_k);
        int n_ok = 0;
        for (int k = 0; k < fd_k; ++k) {
            // dense random unit direction: fd signal is <U,u> ~ ||U|| (large), while the
            // fp32 dot noise stays constant -> high-SNR global check of the whole grad
            double nrm = 0;
            for (auto & x : dir) { x = gauss(frng); nrm += (double) x * x; }
            nrm = sqrt(nrm);
            for (auto & x : dir) x = (float) (x / nrm);

            double u_dot = 0; // <U, u>
            for (size_t i = 0; i < dir.size(); ++i) u_dot += (double) u_host[i] * dir[i];

            auto dot_at = [&](float eps_val) {
                for (size_t i = 0; i < pbuf.size(); ++i) pbuf[i] = bin[i] + eps_val * dir[i];
                ggml_backend_tensor_set(blk.h_in, pbuf.data(), 0, pbuf.size() * sizeof(float));
                if (ggml_backend_graph_compute(backend, gb2) != GGML_STATUS_SUCCESS) die("block compute failed");
                ggml_backend_tensor_get(blk.h_out, h_host.data(), 0, h_host.size() * sizeof(float));
                double dot = 0;
                for (size_t i = 0; i < v_host.size(); ++i) dot += (double) v_host[i] * h_host[i];
                return dot;
            };
            printf("  dir %d: <U,u>=%+.6e\n", k, u_dot);
            const float es[4] = {60.0f, 30.0f, 15.0f, 7.5f};
            double fds[4];
            for (int j = 0; j < 4; ++j) {
                fds[j] = (dot_at(es[j]) - dot_at(-es[j])) / (2.0 * es[j]);
                printf("      eps=%7.2f  fd=%+.6e  rel=%.3e\n", es[j], fds[j],
                       fabs(fds[j] - u_dot) / (fabs(u_dot) + 1e-9));
            }
            // Richardson: central-difference error is O(eps^2), so (4*fd(e/2)-fd(e))/3
            // cancels the leading truncation term; fp32 noise remains (~1/eps)
            bool ok = false;
            for (int j = 0; j < 3; ++j) {
                const double rich = (4.0 * fds[j + 1] - fds[j]) / 3.0;
                const double rel = fabs(rich - u_dot) / (fabs(u_dot) + 1e-9);
                printf("      rich(%5.1f,%5.1f) = %+.6e  rel=%.3e %s\n", es[j], es[j + 1], rich, rel,
                       rel < 3e-2 ? "OK" : "");
                if (rel < 3e-2) ok = true;
            }
            if (ok) n_ok++;
        }
        printf("fd-block %d: %d/%d within 3%%\n", fd_block, n_ok, fd_k);
        return n_ok == fd_k ? 0 : 1;
    }


    // finite-difference validation of the backward pass (same backend, same graph):
    //   fd = (V.h_L(a + eps*e) - V.h_L(a - eps*e)) / (2*eps)  vs  U_L = d(V.h_L)/d(a_L)
    if (fd_layer >= 0) {
        for (auto & x : v_host) x = gauss(rng);
        ggml_backend_tensor_set(vjp_seed, v_host.data(), 0, v_host.size() * sizeof(float));
        std::vector<float> zero(u_host.size(), 0.0f);
        auto print_tokens = [&](const char * tag) {
            std::vector<int32_t> chk(ids.size());
            ggml_backend_tensor_get(fwd.tokens, chk.data(), 0, chk.size() * sizeof(int32_t));
            fprintf(stderr, "DBG tokens @%s: %d %d %d %d %d\n", tag, chk[0], chk[1], chk[2], chk[3], chk[4]);
        };
        print_tokens("pre-zero");
        ggml_backend_tensor_set(fwd.perturb, zero.data(), 0, zero.size() * sizeof(float));
        print_tokens("post-zero");

        if (ggml_backend_graph_compute(backend, gb) != GGML_STATUS_SUCCESS) die("compute failed");
        print_tokens("post-compute");
        ggml_backend_tensor_get(u_t[fd_layer], u_host.data(), 0, u_host.size() * sizeof(float));

        std::vector<float> h_host(u_host.size());
        std::vector<float> pbuf(u_host.size());
        std::mt19937 frng(7);
        std::uniform_int_distribution<int64_t> pick(0, (int64_t) u_host.size() - 1);
        fprintf(stderr, "DBG tokens=%p pos=%p mask=%p perturb=%p seed=%p\n",
                fwd.tokens->data, fwd.pos->data, fwd.mask->data, fwd.perturb->data, vjp_seed->data);
        printf("fd-check layer %d, eps=%g, %d entries (VJP vs central difference)\n", fd_layer, (double) fd_eps, fd_k);
        for (int k = 0; k < fd_k; ++k) {
            const int64_t idx = pick(frng);
            // eps sweep: noise floor (same-perturb repeat) + convergence as eps -> 0
            auto dot_at = [&](float eps_val) {
                std::fill(pbuf.begin(), pbuf.end(), 0.0f);
                pbuf[idx] = eps_val;
                ggml_backend_tensor_set(fwd.perturb, pbuf.data(), 0, pbuf.size() * sizeof(float));
                if (ggml_backend_graph_compute(backend, gb) != GGML_STATUS_SUCCESS) die("compute failed");
                ggml_backend_tensor_get(fwd.l_out.back(), h_host.data(), 0, h_host.size() * sizeof(float));
                double dot = 0;
                for (size_t i = 0; i < v_host.size(); ++i) dot += (double) v_host[i] * h_host[i];
                return dot;
            };
            const double d0a = dot_at(0.0f), d0b = dot_at(0.0f);
            printf("  dim=%5ld pos=%ld  VJP=%+.6e  determinism |d0a-d0b|=%.3e\n",
                   (long)(idx % n_embd), (long)(idx / n_embd), u_host[idx], fabs(d0a - d0b));
            for (float e : {2.0f, 0.5f, 0.125f, 0.03125f}) {
                const double fd = (dot_at(e) - dot_at(-e)) / (2.0 * e);
                printf("      eps=%8.4f  fd=%+.6e  rel=%.3e\n", e, fd,
                       fabs(fd - u_host[idx]) / (fabs(u_host[idx]) + 1e-9));
            }
        }
        return 0;
    }


    // accumulators (only when fitting)
    std::vector<std::vector<float>> M;
    std::vector<float> p_host; // staging for one [n_embd, n_embd] product download
    uint64_t acc_probes = 0, acc_positions = 0;
    const bool fitting = !out_path.empty();
    if (fitting) {
        p_host.resize((size_t) n_embd * n_embd);
        if (!in_path.empty()) {
            if (!acc_load(in_path.c_str(), M, n_embd, acc_probes, acc_positions))
                die("failed to load --in accumulator");
            fprintf(stderr, "jlens-fit: continuing from %s (%llu probes, %llu positions)\n",
                    in_path.c_str(), (unsigned long long) acc_probes, (unsigned long long) acc_positions);
        } else {
            M.assign(n_layer, std::vector<float>((size_t) n_embd * n_embd, 0.0f));
        }
        fprintf(stderr, "jlens-fit: accumulators %.2f GiB host RAM\n",
                (double) n_layer * n_embd * n_embd * 4 / (1024.0 * 1024.0 * 1024.0));
    }

    // dump-grads reference loading happens after probe 0 compute

    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        if (!tokens_file.empty()) {
            ggml_backend_tensor_set(fwd.tokens, ids.data() + (size_t) chunk * stride, 0,
                                    n_tokens * sizeof(int32_t));
        }
        if (lrp) {
            const int64_t th0 = ggml_time_us();
            harvest();
            fprintf(stderr, "chunk %d/%d: harvest %.1f ms\n",
                    chunk + 1, n_chunks, (ggml_time_us() - th0) / 1000.0);
        }
    for (int probe = 0; probe < n_probes; ++probe) {
        for (auto & x : v_host) x = gauss(rng);
        ggml_backend_tensor_set(vjp_seed, v_host.data(), 0, v_host.size() * sizeof(float));

        const int64_t t0 = ggml_time_us();
        if (ggml_backend_graph_compute(backend, gb) != GGML_STATUS_SUCCESS) die("compute failed");
        const int64_t t1 = ggml_time_us();

        if (probe == 0 && (!dump_path.empty() || !compare_path.empty())) {
            FILE * f = nullptr;
            std::vector<float> ref;
            if (!dump_path.empty()) {
                f = fopen(dump_path.c_str(), "wb");
                if (!f) die("cannot open dump path");
                const uint32_t hdr[3] = { (uint32_t) n_layer, (uint32_t) n_embd, (uint32_t) n_tokens };
                fwrite(hdr, 4, 3, f);
            } else {
                f = fopen(compare_path.c_str(), "rb");
                if (!f) die("cannot open compare path");
                uint32_t hdr[3];
                if (fread(hdr, 4, 3, f) != 3) die("bad compare file");
                if (hdr[0] != (uint32_t) n_layer || hdr[1] != (uint32_t) n_embd || hdr[2] != (uint32_t) n_tokens)
                    die("compare file shape mismatch");
                ref.resize((size_t) n_embd * n_tokens);
            }
            double worst_rel = 0.0;
            for (int il = 0; il < n_layer; ++il) {
                ggml_backend_tensor_get(u_t[il], u_host.data(), 0, u_host.size() * sizeof(float));
                if (!dump_path.empty()) {
                    fwrite(u_host.data(), sizeof(float), u_host.size(), f);
                } else {
                    if (fread(ref.data(), sizeof(float), ref.size(), f) != ref.size()) die("short compare file");
                    double max_abs = 0.0, max_ref = 0.0;
                    size_t argmax = 0;
                    int n_bad = 0;
                    for (size_t k = 0; k < ref.size(); ++k) {
                        const double d = fabs(u_host[k] - ref[k]);
                        const double r = fabs(ref[k]);
                        if (d > max_abs) { max_abs = d; argmax = k; }
                        max_ref = std::max(max_ref, r);
                        if (d > 0.01 * (r + 1e-3)) n_bad++;
                    }
                    const double rel = max_abs / (max_ref + 1e-12);
                    worst_rel = std::max(worst_rel, rel);
                    if (il % 10 == 0 || il == n_layer - 1 || (il < 3 && max_abs > 0))
                        printf("layer %2d: max|dU|=%.3e max|U|=%.3e rel=%.3e n_bad=%d/%zu argmax dim=%ld pos=%ld (U=%.4f ref=%.4f)\n",
                               il, max_abs, max_ref, rel, n_bad, ref.size(),
                               (long)(argmax % n_embd), (long)(argmax / n_embd),
                               u_host[argmax], ref[argmax]);
                }
            }
            fclose(f);
            if (!dump_path.empty()) printf("dumped %d layer VJPs to %s\n", (int) n_layer, dump_path.c_str());
            else printf("worst relative difference: %.3e %s\n", worst_rel,
                        worst_rel < 1e-3 ? "(OK)" : "(TOO LARGE - investigate)");
            return 0;
        }

        if (fitting) {
            // fetch VJPs and accumulate M_l += V U_l^T per layer
            if (!acc_on_gpu) memcpy(av->data, v_host.data(), v_host.size() * sizeof(float));
            for (int il = 0; il < n_layer; ++il) {
                if (acc_on_gpu) {
                    au->data   = u_t[il]->data;
                    au->buffer = u_t[il]->buffer;
                    if (ggml_backend_graph_compute(acc_backend, acc_gf) != GGML_STATUS_SUCCESS) die("acc compute failed");
                    ggml_backend_tensor_get(aP, p_host.data(), 0, p_host.size() * sizeof(float));
                    acc_add(M[il], p_host.data(), n_embd, n_threads);
                } else {
                    ggml_backend_tensor_get(u_t[il], u_host.data(), 0, u_host.size() * sizeof(float));
                    memcpy(au->data, u_host.data(), u_host.size() * sizeof(float));
                    if (ggml_backend_graph_compute(acc_backend, acc_gf) != GGML_STATUS_SUCCESS) die("acc compute failed");
                    acc_add(M[il], (const float *) aP->data, n_embd, n_threads);
                }
            }
            acc_probes    += 1;
            acc_positions += n_tokens;
        }
        const int64_t t2 = ggml_time_us();
        fprintf(stderr, "chunk %d/%d probe %d/%d: compute %.1f ms, accumulate %.1f ms\n",
                chunk + 1, n_chunks, probe + 1, n_probes, (t1 - t0) / 1000.0, (t2 - t1) / 1000.0);
    }
        if (fitting && save_every > 0 && (chunk + 1) % save_every == 0) {
            if (!acc_save(out_path.c_str(), M, n_embd, acc_probes, acc_positions))
                die("failed to checkpoint accumulators");
            fprintf(stderr, "jlens-fit: checkpoint at chunk %d (%llu probes, %llu positions)\n",
                    chunk + 1, (unsigned long long) acc_probes, (unsigned long long) acc_positions);
        }
    }

    if (fitting) {
        if (!acc_save(out_path.c_str(), M, n_embd, acc_probes, acc_positions)) die("failed to save accumulators");
        printf("saved accumulators (%llu probes, %llu positions) to %s\n",
               (unsigned long long) acc_probes, (unsigned long long) acc_positions, out_path.c_str());
    }
    return 0;
}
