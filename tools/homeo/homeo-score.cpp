// homeo-score: score-function identification of token-mediated regulation (plan B2 in NOTES-homeostasis.md;
// "Proposal: measuring push-back without pushing").
//
// homeo-ident pushes the block-L output by c*u from the last prefix token on and measures how the later layer-L
// states respond through the written tokens. The same response can be read off unpushed text: the push changes
// the later states only by changing which tokens get written, so for any readout r of the pre-push states
//   d E[r] / d c = E[(r - b) S],   S = d log p(written tokens) / d c = G . u,
//   G = sum over the pushed positions s of d log p(written tokens) / d h_s   (h_s = block-L output)
// with b any baseline that does not depend on the written tokens (likelihood ratio). With R the readout and
// G stored as full vectors, the response matrix in any basis Q is M_jk = E[(q_j . R - b_j)(q_k . G)].
//
// Per prompt: the prefix [0, plen - 1) is decoded once (llama.cpp, for generation; and the jspace chunk forward,
// for its keys/values). Continuations are sampled in parallel at T=1, untruncated, with an independent random
// stream per continuation (seeded by prompt and continuation index, so a rerun reproduces them). Lag k is
// position plen - 1 + k: lag 0 is the last prefix token, lag k >= 1 has written token k - 1 as input, and the
// written token k is predicted at lag k. The pushed positions are lags 0 .. n_gen - 1, as in homeo-ident.
// G comes from one backward pass through blocks L+1 .. n_layer-1 and the head (jspace graphs, explicit f32
// attention; the prefix keys/values fixed), batched over --grad-batch continuations with a block-diagonal mask.
// No push is ever applied: under the welfare protocol this is ordinary sampling plus gradients (no exposure).
//
// Output (resumable): OUT.bin, fixed-size records in order (prompt, continuation):
//   i32 prompt, i32 cont, i32 tok[n_gen], f32 logp[n_gen] (llama.cpp log p of each written token),
//   f32 ent[n_gen] (entropy of each next-token distribution), f32 G[n_embd],
//   f32 R[n_win][n_embd] (mean block-L output over each lag window)
// OUT.meta.json describes the run. A rerun with the same arguments appends the missing records.
// --layers SPEC (e.g. "all", "0-60", "24,32,40-44"): capture several layers in one run. The backward pass then runs
// from the head down to the lowest one in chained groups of at most --max-group blocks (the gradient at each group's
// bottom is fed to the group below as its top gradient), and the gradient at every block output inside a group is
// kept, so each listed layer L gets its own G_L (push after block L, summed over lags 0 .. n_gen-1) and readouts R_L.
// Records then hold, after the per-token fields, f16 G[n_embd] and f16 R[n_win][n_embd] for each listed layer, in
// order (meta "layers", "dtype": "f16"). Without --layers: one layer (--layer), f32, the layout above.
// --check N: for the first N continuations of the first prompt still to do, compare G and the readout states
// with jlens_value_exact on the whole text (prefix + written tokens), which was validated against finite
// differences (VL5), and compare G between two computes of the same batch.
//
// usage: homeo-score -m model.gguf --ids ids.bin --out PREFIX [--layer 40] [--n-prompt 32] [--prefix-len 256]
//          [--n-gen 32] [--n-cont 320] [--n-par 32] [--grad-batch 4] [--windows 1-4,5-12,13-31]
//          [--ids-offset 500000] [--ids-stride 100003] [--seed 1] [--pause-ms 0] [--check 0] [--prefix-chunk 0]
//          [--threads N]
// --offsets FILE: prompt windows start at these stream offsets (one per line; overrides --ids-offset/--ids-stride;
// --n-prompt defaults to the number of lines).
// --prefix-chunk C: decode the prefix keys/values in chunks of C tokens (0 = one chunk). With C = 32 and
// --grad-batch 1 the matmul shapes match jlens_value_exact's 32-token chunks (for --check).
// --pause-ms is re-read from PREFIX.pause_ms (if it exists) before every batch, to change the duty cycle live.

#include "jlens.h"

#include "llama.h"
#include "src/llama-model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <csignal>
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

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

static void die(const char * msg) {
    fprintf(stderr, "homeo-score: %s\n", msg);
    exit(1);
}

static double now_sec() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class F>
static void parallel_for(int n, int n_threads, const F & f) {
    const int chunk = (n + n_threads - 1) / n_threads;
    std::vector<std::thread> th;
    for (int t = 0; t < n_threads; ++t) {
        const int a = t * chunk, b = std::min(n, a + chunk);
        if (a >= b) break;
        th.emplace_back([a, b, &f] { for (int i = a; i < b; ++i) f(i); });
    }
    for (auto & x : th) x.join();
}

static long long file_size(const std::string & p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 ? (long long) st.st_size : -1;
}

// point a leaf at existing device memory (gallocr skips tensors that already have data)
static void wire(ggml_tensor * leaf, const ggml_tensor * src, size_t offset) {
    leaf->data   = (char *) src->data + offset;
    leaf->buffer = src->buffer;
}

// the llama.cpp model's weights through the jlens_weights interface (as server-jlens.cpp)
struct llama_jlens_weights : jlens_weights {
    std::unordered_map<std::string, ggml_tensor *> map;

    explicit llama_jlens_weights(const llama_model * lm) {
        for (const auto & p : lm->tensors_by_name) map[p.first] = p.second;
        const auto & hp = lm->hparams;
        hparams.n_embd         = hp.n_embd;
        hparams.n_layer        = hp.n_layer();
        hparams.n_head         = hp.n_head();
        hparams.n_head_kv      = hp.n_head_kv();
        hparams.n_embd_head    = hp.n_embd_head_k();
        hparams.n_ff           = hp.n_ff();
        hparams.n_rot          = hp.n_rot();
        hparams.n_ctx_train    = hp.n_ctx_train;
        hparams.rope_freq_base = hp.rope_freq_base_train;
        hparams.norm_eps       = hp.f_norm_rms_eps;
        hparams.n_vocab        = get("output.weight")->ne[1];
    }

    ggml_tensor * get(const std::string & name) const override {
        auto it = map.find(name);
        return it == map.end() ? nullptr : it->second;
    }
};

// device stores shared by the graphs: keys/values of every block over [0, n_kv) (the prefix at [0, plen - 1),
// the rest is overwritten in-graph by the chunk's own), the block outputs of the gradient batch that are needed
// (captured layers and group bottoms), the gradient handed from each backward group to the one below, zeros, 1.0
struct stores {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;
    std::vector<ggml_tensor *> h;  // [n_layer]: [n_embd, n_tok] if needed, else nullptr
    std::vector<ggml_tensor *> g;  // [n_groups]: gradient at group gi's bottom, [n_embd, n_tok]
    ggml_tensor * zero = nullptr;  // [D, Hkv, n_tok]
    ggml_tensor * seed = nullptr;  // [1] = 1.0

    void alloc(const jlens_hparams & hp, ggml_backend_buffer_type_t buft, int64_t n_kv, int n_tok, const std::vector<int> & need, int n_g) {
        ggml_init_params ip = { (size_t) (3 * hp.n_layer + n_g + 8) * ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        for (int il = 0; il < hp.n_layer; ++il) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.n_embd_head, hp.n_head_kv, n_kv));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.n_embd_head, hp.n_head_kv, n_kv));
        }
        h.assign(hp.n_layer, nullptr);
        for (int il : need) h[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, n_tok);
        for (int gi = 0; gi < n_g; ++gi) g.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, n_tok));
        zero = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp.n_embd_head, hp.n_head_kv, n_tok);
        seed = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) die("cannot allocate the device stores");
        ggml_backend_buffer_clear(buf, 0);
        const float one = 1.0f;
        ggml_backend_tensor_set(seed, &one, 0, sizeof(float));
    }
};

// one backward group (lo, hi]: blocks lo+1 .. hi from the stored block-lo output, with the head (hi = last block)
// or the gradient at the hi output from the group above; the gradients kept for the captured layers in [lo, hi)
// (and hi itself on the head group)
struct bgroup {
    int lo = 0, hi = 0;
    jlens_upper_chunk uc;
    ggml_cgraph * gb = nullptr;
    ggml_tensor * gh = nullptr;                 // d loss / d (block-lo output)
    std::vector<std::pair<int, ggml_tensor *>> caps; // (index into the layer list, gradient tensor [n_embd, n_tok])
};

int main(int argc, char ** argv) {
    std::string model_path, ids_path, out_prefix, windows_arg = "1-4,5-12,13-31", offsets_path, layers_arg;
    int max_group = 21;
    int layer = 40, n_prompt = 32, plen = 256, G = 32, n_cont = 320, n_par = 32, gb = 4, pause_ms = 0, n_check = 0, pchunk = 0;
    int n_threads = std::max(1, (int) std::thread::hardware_concurrency() / 2);
    long long ids_offset = 500000, ids_stride = 100003;
    uint64_t seed = 1;
    bool n_prompt_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) die(("missing value for " + a).c_str()); return argv[++i]; };
        if      (a == "-m")             model_path = next();
        else if (a == "--ids")          ids_path   = next();
        else if (a == "--out")          out_prefix = next();
        else if (a == "--layer")        layer      = atoi(next());
        else if (a == "--n-prompt")   { n_prompt   = atoi(next()); n_prompt_given = true; }
        else if (a == "--prefix-len")   plen       = atoi(next());
        else if (a == "--n-gen")        G          = atoi(next());
        else if (a == "--n-cont")       n_cont     = atoi(next());
        else if (a == "--n-par")        n_par      = atoi(next());
        else if (a == "--grad-batch")   gb         = atoi(next());
        else if (a == "--windows")      windows_arg = next();
        else if (a == "--ids-offset")   ids_offset = atoll(next());
        else if (a == "--ids-stride")   ids_stride = atoll(next());
        else if (a == "--seed")         seed       = strtoull(next(), nullptr, 10);
        else if (a == "--pause-ms")     pause_ms   = atoi(next());
        else if (a == "--check")        n_check    = atoi(next());
        else if (a == "--prefix-chunk") pchunk     = atoi(next());
        else if (a == "--offsets")      offsets_path = next();
        else if (a == "--layers")       layers_arg = next();
        else if (a == "--max-group")    max_group  = atoi(next());
        else if (a == "--threads")      n_threads  = atoi(next());
        else die(("unknown argument " + a).c_str());
    }
    if (model_path.empty() || ids_path.empty() || out_prefix.empty()) {
        fprintf(stderr, "usage: homeo-score -m model.gguf --ids ids.bin --out PREFIX [options]\n");
        return 1;
    }
    if (n_par % gb != 0 || n_cont % n_par != 0) die("--n-par must be a multiple of --grad-batch, and --n-cont of --n-par");

    // lag windows "a-b,c-d"
    std::vector<std::pair<int, int>> wins;
    {
        size_t p = 0;
        while (p < windows_arg.size()) {
            const size_t q = std::min(windows_arg.find(',', p), windows_arg.size());
            const std::string w = windows_arg.substr(p, q - p);
            const size_t d = w.find('-');
            if (d == std::string::npos) die("bad --windows");
            wins.push_back({ atoi(w.substr(0, d).c_str()), atoi(w.substr(d + 1).c_str()) });
            if (wins.back().first < 0 || wins.back().second >= G || wins.back().first > wins.back().second) die("bad --windows");
            p = q + 1;
        }
    }
    const int NW = (int) wins.size();

    // ---- model: llama.cpp, every tensor on CUDA0 (token embeddings too, for the jspace graphs) ----
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    const llama_model_tensor_buft_override ovr[] = { { "token_embd\\.weight", ggml_backend_cuda_buffer_type(0) }, { nullptr, nullptr } };
    mp.tensor_buft_overrides = ovr;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) die("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_jlens_weights W(model);
    const auto & hp = W.hparams;
    const int n_embd = (int) hp.n_embd, n_layer = (int) hp.n_layer;
    // captured layers, ascending
    std::vector<int> LY;
    const bool multi = !layers_arg.empty();
    if (!multi) {
        if (layer < 0 || layer >= n_layer - 1) die("--layer must be in [0, n_layer - 1)");
        LY = { layer };
    } else if (layers_arg == "all") {
        for (int il = 0; il < n_layer; ++il) LY.push_back(il);
    } else {
        size_t p = 0;
        while (p < layers_arg.size()) {
            const size_t q = std::min(layers_arg.find(',', p), layers_arg.size());
            const std::string t = layers_arg.substr(p, q - p);
            const size_t d = t.find('-');
            const int l0 = atoi(t.substr(0, d).c_str()), l1 = d == std::string::npos ? l0 : atoi(t.substr(d + 1).c_str());
            for (int il = l0; il <= l1; ++il) LY.push_back(il);
            p = q + 1;
        }
        std::sort(LY.begin(), LY.end());
        LY.erase(std::unique(LY.begin(), LY.end()), LY.end());
    }
    for (int il : LY) if (il < 0 || il >= n_layer) die("--layers: layer out of range");
    const int nL = (int) LY.size();
    if (max_group < 2) die("--max-group must be >= 2");
    // backward groups, top down: (lo, hi] with the head on the first; the lowest lo is the lowest captured layer
    std::vector<int> bounds = { n_layer - 1 };
    while (bounds.back() > LY.front()) bounds.push_back(std::max(LY.front(), bounds.back() - max_group));
    if (bounds.size() == 1) bounds.push_back(n_layer - 2); // only the last block captured: one group (n_layer-2, n_layer-1]
    const int n_groups = (int) bounds.size() - 1;
    // block outputs needed on the device: captured layers and group bottoms
    std::vector<int> need = LY;
    for (int gi = 0; gi < n_groups; ++gi) need.push_back(bounds[gi + 1]);
    std::sort(need.begin(), need.end());
    need.erase(std::unique(need.begin(), need.end()), need.end());
    for (const auto & kv : W.map) {
        if (!kv.second->buffer || strncmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(kv.second->buffer)), "CUDA0", 5) != 0)
            die(("tensor not on CUDA0: " + kv.first).c_str());
    }

    // ---- prompts: windows of a token-id stream ----
    std::vector<std::vector<int32_t>> pre;
    {
        std::ifstream f(ids_path, std::ios::binary | std::ios::ate);
        if (!f) die("cannot open --ids");
        std::vector<int32_t> ids((size_t) f.tellg() / sizeof(int32_t));
        f.seekg(0);
        f.read((char *) ids.data(), ids.size() * sizeof(int32_t));
        std::vector<long long> offs;
        if (!offsets_path.empty()) {
            std::ifstream fo(offsets_path);
            long long o;
            while (fo >> o) offs.push_back(o);
            if (offs.empty()) die("--offsets: no offsets read");
            if (n_prompt > (int) offs.size() || !n_prompt_given) n_prompt = (int) offs.size();
        }
        for (int i = 0; i < n_prompt; ++i) {
            const long long o = offs.empty() ? ids_offset + (long long) i * ids_stride : offs[i];
            if (o + plen > (long long) ids.size()) die("--ids stream too short for --n-prompt windows");
            pre.emplace_back(ids.begin() + o, ids.begin() + o + plen);
        }
    }

    // ---- output, resume ----
    const size_t rec_ints   = 2 + (size_t) G;
    const size_t rec_floats = 2 * (size_t) G;
    const size_t rec_vec    = (size_t) nL * n_embd * (1 + NW); // G and R per layer
    const size_t rec_bytes  = rec_ints * sizeof(int32_t) + rec_floats * sizeof(float) + rec_vec * (multi ? sizeof(ggml_fp16_t) : sizeof(float));
    const std::string bin_path = out_prefix + ".bin";
    long long have = std::max(0LL, file_size(bin_path));
    const long long total = (long long) n_prompt * n_cont;
    {
        json meta;
        meta["tool"] = "homeo-score"; meta["model"] = model_path; meta["ids"] = ids_path;
        meta["ids_offset"] = ids_offset; meta["ids_stride"] = ids_stride; meta["offsets"] = offsets_path;
        if (!multi) meta["layer"] = layer;
        meta["n_prompt"] = n_prompt; meta["prefix_len"] = plen; meta["n_gen"] = G; meta["n_cont"] = n_cont;
        meta["n_embd"] = n_embd; meta["windows"] = json::array();
        for (auto & w : wins) meta["windows"].push_back({ w.first, w.second });
        meta["seed"] = seed; meta["record_bytes"] = rec_bytes;
        if (multi) {
            meta["layers"] = LY; meta["dtype"] = "f16"; meta["max_group"] = max_group;
            meta["layout"] = "i32 prompt, i32 cont, i32 tok[n_gen], f32 logp[n_gen], f32 ent[n_gen], then per layer: f16 G[n_embd], f16 R[n_win][n_embd]";
        } else {
            meta["layout"] = "i32 prompt, i32 cont, i32 tok[n_gen], f32 logp[n_gen], f32 ent[n_gen], f32 G[n_embd], f32 R[n_win][n_embd]";
        }
        meta["lag"] = "lag k = position prefix_len - 1 + k; pushed positions = lags 0 .. n_gen-1; G = d log p(written tokens) / d h summed over them";
        if (have > 0) {
            // appending: the existing records must come from the same configuration
            std::ifstream fm(out_prefix + ".meta.json");
            if (!fm) die("OUT.bin exists but OUT.meta.json does not");
            json old = json::parse(fm);
            for (const char * key : { "model", "ids", "ids_offset", "ids_stride", "offsets", "layer", "layers", "n_prompt", "prefix_len",
                                      "n_gen", "n_cont", "n_embd", "windows", "seed", "record_bytes", "dtype" }) {
                if (old.contains(key) != meta.contains(key) || (meta.contains(key) && old[key] != meta[key]))
                    die((std::string("cannot resume: ") + key + " differs from the existing run (OUT.meta.json)").c_str());
            }
            if (have % (long long) rec_bytes != 0) {
                const long long whole = have - have % (long long) rec_bytes;
                fprintf(stderr, "homeo-score: OUT.bin ends in a partial record; truncating %lld -> %lld bytes\n", have, whole);
                if (truncate(bin_path.c_str(), (off_t) whole) != 0) die("cannot truncate OUT.bin");
                have = whole;
            }
        }
        std::ofstream(out_prefix + ".meta.json") << meta.dump(1) << "\n";
    }
    long long done = have / (long long) rec_bytes;
    FILE * fout = fopen(bin_path.c_str(), "ab");
    if (!fout) die("cannot open OUT.bin");
    fprintf(stderr, "homeo-score: %d layer(s) %d..%d, %d backward group(s), %d prompts x %d continuations of %d tokens, %lld of %lld records done\n",
            nL, LY.front(), LY.back(), n_groups, n_prompt, n_cont, G, done, total);

    // ---- llama.cpp context for generation ----
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) ((n_par + 1) * (plen + G) + 512);
    cp.n_batch         = 512;
    cp.n_ubatch        = 512;
    cp.n_seq_max       = n_par + 1; // 0 .. n_par-1 continuations, n_par the shared prefix
    cp.kv_unified      = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.no_perf         = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context init failed");
    llama_memory_t mem = llama_get_memory(ctx);
    llama_batch batch = llama_batch_init(std::max(512, n_par), 0, 1);

    // ---- jspace graphs ----
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) die("cuda backend init failed");
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(0);
    const int64_t P0 = plen - 1;            // prefix positions with fixed keys/values
    const int     NT = gb * G;              // tokens per gradient batch
    const int64_t NKV = P0 + NT;
    const int64_t D = hp.n_embd_head, Hkv = hp.n_head_kv;
    stores st;
    st.alloc(hp, buft, NKV, NT, need, n_groups);

    // (1) prefix forward (per prompt, below): keys/values of [0, P0) for every block, in chunks of pc tokens
    const int pc = pchunk > 0 ? std::min<int>(pchunk, (int) P0) : (int) P0;
    ggml_gallocr_t ga_pf = ggml_gallocr_new(buft);
    auto prefix_forward = [&](const std::vector<int32_t> & pr) {
        std::vector<int32_t> posv;
        std::vector<float> mask;
        for (int64_t a = 0; a < P0; a += pc) {
            const int64_t b = std::min<int64_t>(P0, a + pc);
            const int c = (int) (b - a);
            jlens_chunk_forward pf;
            if (!jlens_build_chunk_forward(W, a, c, b, pf)) die("prefix graph build failed");
            wire(pf.one, st.seed, 0);
            for (int il = 0; il < n_layer; ++il) {
                wire(pf.k_kv[il], st.k[il], 0);
                wire(pf.v_kv[il], st.v[il], 0);
                for (int kv = 0; kv < 2; ++kv) {
                    ggml_tensor * dst = ggml_new_tensor_3d(pf.gctx, GGML_TYPE_F32, D, Hkv, c);
                    wire(dst, kv == 0 ? st.k[il] : st.v[il], (size_t) a * st.k[il]->nb[2]);
                    ggml_build_forward_expand(pf.gf, ggml_cpy(pf.gctx, kv == 0 ? pf.k_live[il] : pf.v_live[il], dst));
                }
            }
            if (!ggml_gallocr_alloc_graph(ga_pf, pf.gf)) die("prefix graph alloc failed (out of VRAM?)");
            posv.resize(c);
            for (int i = 0; i < c; ++i) posv[i] = (int32_t) (a + i);
            mask.assign((size_t) b * c, 0.0f);
            for (int i = 0; i < c; ++i)
                for (int64_t j = a + i + 1; j < b; ++j) mask[(size_t) j + (size_t) i * b] = -FLT_MAX;
            ggml_backend_tensor_set(pf.tokens, pr.data() + a, 0, c * sizeof(int32_t));
            ggml_backend_tensor_set(pf.pos, posv.data(), 0, c * sizeof(int32_t));
            ggml_backend_tensor_set(pf.mask, mask.data(), 0, mask.size() * sizeof(float));
            const bool ok = ggml_backend_graph_compute(backend, pf.gf) == GGML_STATUS_SUCCESS;
            ggml_free(pf.gctx);
            if (!ok) die("prefix forward failed");
        }
    };

    // (2) chunk forward of the gradient batch through blocks 0 .. max(need): the needed block outputs into st.h
    jlens_chunk_forward cf;
    if (!jlens_build_chunk_forward(W, P0, NT, NKV, cf)) die("chunk graph build failed");
    wire(cf.one, st.seed, 0);
    for (int il = 0; il < n_layer; ++il) {
        wire(cf.k_kv[il], st.k[il], 0);
        wire(cf.v_kv[il], st.v[il], 0);
    }
    for (int il : need) {
        ggml_tensor * dst = ggml_new_tensor_2d(cf.gctx, GGML_TYPE_F32, n_embd, NT);
        wire(dst, st.h[il], 0);
        ggml_build_forward_expand(cf.gf, ggml_cpy(cf.gctx, cf.l_out[il], dst));
    }
    ggml_gallocr_t ga_cf = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(ga_cf, cf.gf)) die("chunk graph alloc failed (out of VRAM?)");

    // (3) backward groups, top down: d (sum of NLL of the written tokens) / d (block outputs)
    std::vector<int> li_of(n_layer, -1);
    for (int i = 0; i < nL; ++i) li_of[LY[i]] = i;
    std::vector<bgroup> grp(n_groups);
    for (int gi = 0; gi < n_groups; ++gi) {
        bgroup & g = grp[gi];
        g.hi = bounds[gi]; g.lo = bounds[gi + 1];
        const bool head = gi == 0;
        if (!jlens_build_upper_chunk(W, g.lo, P0, NT, NKV, NT, /*now_mode=*/false, g.uc, g.hi)) die("upper graph build failed");
        wire(g.uc.h, st.h[g.lo], 0);
        wire(g.uc.one, st.seed, 0);
        if (!head) wire(g.uc.g_top, st.g[gi - 1], 0);
        for (int u = 0; u < g.hi - g.lo; ++u) {
            wire(g.uc.k_kv[u], st.k[g.lo + 1 + u], 0);
            wire(g.uc.v_kv[u], st.v[g.lo + 1 + u], 0);
            wire(g.uc.dk_in[u], st.zero, 0);
            wire(g.uc.dv_in[u], st.zero, 0);
        }
        std::vector<ggml_tensor *> grad_accs(ggml_graph_n_nodes(g.uc.gf), nullptr);
        for (int i = 0; i < ggml_graph_n_nodes(g.uc.gf); ++i) {
            if (ggml_graph_node(g.uc.gf, i) == g.uc.loss) grad_accs[i] = st.seed;
        }
        g.gb = ggml_graph_dup(g.uc.gctx, g.uc.gf, /*force_grads=*/true);
        ggml_build_backward_expand(g.uc.gctx, g.gb, grad_accs.data());
        g.gh = ggml_graph_get_grad(g.gb, g.uc.h);
        if (!g.gh) die("no gradient for a group's bottom input");
        ggml_set_output(g.gh);
        if (gi + 1 < n_groups) {
            // hand the gradient at this group's bottom to the group below
            ggml_tensor * dst = ggml_new_tensor_2d(g.uc.gctx, GGML_TYPE_F32, n_embd, NT);
            wire(dst, st.g[gi], 0);
            ggml_build_forward_expand(g.gb, ggml_cpy(g.uc.gctx, g.gh, dst));
        }
        if (li_of[g.lo] >= 0) g.caps.push_back({ li_of[g.lo], g.gh });
        for (int u = 0; u < g.hi - g.lo; ++u) {
            const int il = g.lo + 1 + u;
            if (li_of[il] < 0 || (il == g.hi && !head)) continue; // a non-head group's top = the bottom of the group above
            ggml_tensor * t = ggml_graph_get_grad(g.gb, g.uc.l_out[u]);
            if (!t) die("no gradient for a block output");
            ggml_set_output(t);
            g.caps.push_back({ li_of[il], t });
        }
    }
    // one allocator for all backward groups. ggml_gallocr_alloc_graph skips tensors that already have data, so each
    // group keeps the addresses of its first allocation: the buffer must never be reallocated after that. Reserve
    // every group's graph first (the buffer grows to the largest), then allocate; grad_batch checks the size stays.
    ggml_gallocr_t ga_uc = ggml_gallocr_new(buft);
    for (auto & g : grp) {
        if (!ggml_gallocr_reserve(ga_uc, g.gb)) die("upper graph reserve failed (out of VRAM?): lower --grad-batch or --max-group");
    }
    for (auto & g : grp) {
        if (!ggml_gallocr_alloc_graph(ga_uc, g.gb)) die("upper graph alloc failed (out of VRAM?): lower --grad-batch or --max-group");
    }
    const size_t ga_uc_size = ggml_gallocr_get_buffer_size(ga_uc, 0);
    const int64_t n_lab = grp[0].uc.labels->ne[0];
    {
        int n_caps = 0;
        for (auto & g : grp) n_caps += (int) g.caps.size();
        if (n_caps != nL) die("internal: captured layer count mismatch");
    }
    fprintf(stderr, "homeo-score: graphs ready (gradient batch %d continuations = %d tokens, %lld key slots; groups", gb, NT, (long long) NKV);
    for (auto & g : grp) fprintf(stderr, " (%d,%d]", g.lo, g.hi);
    fprintf(stderr, ")\n");

    // inputs that do not change
    std::vector<int32_t> pos_c(NT);
    for (int b = 0; b < gb; ++b)
        for (int k = 0; k < G; ++k) pos_c[b * G + k] = (int32_t) (P0 + k);
    std::vector<float> mask_c((size_t) NKV * NT, -FLT_MAX);
    for (int b = 0; b < gb; ++b)
        for (int k = 0; k < G; ++k) {
            float * row = mask_c.data() + (size_t) (b * G + k) * NKV;
            for (int64_t j = 0; j < P0; ++j) row[j] = 0.0f;
            for (int j = 0; j <= k; ++j) row[P0 + b * G + j] = 0.0f;
        }
    std::vector<float> labels((size_t) n_lab * NT, 0.0f), gbuf((size_t) n_embd * NT);
    std::vector<std::vector<float>> hbufL(nL, std::vector<float>((size_t) n_embd * NT));

    // one gradient batch: tokens [gb][G] (lag inputs) and next [gb][G] (written tokens) -> per continuation and
    // captured layer, G [gb][nL][n_embd] and R [gb][nL][NW][n_embd]; the block outputs stay in hbufL
    auto grad_batch = [&](const std::vector<std::vector<int32_t>> & in, const std::vector<std::vector<int32_t>> & nx,
                          std::vector<float> & Gout, std::vector<float> & Rout, double & nll) {
        std::vector<int32_t> tk(NT);
        std::fill(labels.begin(), labels.end(), 0.0f);
        for (int b = 0; b < gb; ++b)
            for (int k = 0; k < G; ++k) {
                tk[b * G + k] = in[b][k];
                labels[(size_t) (b * G + k) * n_lab + nx[b][k]] = 1.0f;
            }
        ggml_backend_tensor_set(cf.tokens, tk.data(), 0, NT * sizeof(int32_t));
        ggml_backend_tensor_set(cf.pos, pos_c.data(), 0, NT * sizeof(int32_t));
        ggml_backend_tensor_set(cf.mask, mask_c.data(), 0, mask_c.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, cf.gf) != GGML_STATUS_SUCCESS) die("chunk forward failed");
        Gout.assign((size_t) gb * nL * n_embd, 0.0f);
        Rout.assign((size_t) gb * nL * NW * n_embd, 0.0f);
        for (int li = 0; li < nL; ++li) {
            std::vector<float> & hb = hbufL[li];
            ggml_backend_tensor_get(st.h[LY[li]], hb.data(), 0, hb.size() * sizeof(float));
            for (int b = 0; b < gb; ++b)
                for (int w = 0; w < NW; ++w) {
                    float * r = Rout.data() + (((size_t) b * nL + li) * NW + w) * n_embd;
                    const int k0 = wins[w].first, k1 = wins[w].second;
                    for (int k = k0; k <= k1; ++k) {
                        const float * src = hb.data() + (size_t) (b * G + k) * n_embd;
                        for (int e = 0; e < n_embd; ++e) r[e] += src[e];
                    }
                    for (int e = 0; e < n_embd; ++e) r[e] /= (float) (k1 - k0 + 1);
                }
        }
        nll = 0.0;
        for (int gi = 0; gi < n_groups; ++gi) {
            bgroup & g = grp[gi];
            if (n_groups > 1 && !ggml_gallocr_alloc_graph(ga_uc, g.gb)) die("upper graph alloc failed (out of VRAM?): lower --grad-batch or --max-group");
            if (ggml_gallocr_get_buffer_size(ga_uc, 0) != ga_uc_size) die("internal: backward buffer was reallocated");
            ggml_backend_tensor_set(g.uc.pos, pos_c.data(), 0, NT * sizeof(int32_t));
            ggml_backend_tensor_set(g.uc.mask, mask_c.data(), 0, mask_c.size() * sizeof(float));
            if (gi == 0) ggml_backend_tensor_set(g.uc.labels, labels.data(), 0, labels.size() * sizeof(float));
            if (ggml_backend_graph_compute(backend, g.gb) != GGML_STATUS_SUCCESS) die("backward failed");
            if (gi == 0) {
                float loss = 0.0f;
                ggml_backend_tensor_get(g.uc.loss, &loss, 0, sizeof(float));
                nll = loss;
            }
            for (auto & c : g.caps) {
                ggml_backend_tensor_get(c.second, gbuf.data(), 0, gbuf.size() * sizeof(float));
                for (int b = 0; b < gb; ++b) {
                    float * gout = Gout.data() + ((size_t) b * nL + c.first) * n_embd;
                    for (int k = 0; k < G; ++k) {
                        const float * src = gbuf.data() + (size_t) (b * G + k) * n_embd;
                        for (int e = 0; e < n_embd; ++e) gout[e] -= src[e]; // loss = NLL, so d log p = -d loss
                    }
                }
            }
        }
    };

    // --check works at one layer: --layer if captured, else the highest captured layer below the last block
    int lc = li_of[layer];
    if (lc < 0) for (int i = nL - 1; i >= 0 && lc < 0; --i) if (LY[i] < n_layer - 1) lc = i;
    if (lc < 0) lc = 0;
    const int Lc = LY[lc];
    if (n_check > 0) fprintf(stderr, "homeo-score: checks at layer %d\n", Lc);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::vector<std::vector<float>> lp(n_par, std::vector<float>(n_vocab));
    const int n_batches = n_cont / n_par;
    double t_gen = 0, t_grad = 0;
    long long n_written = 0;
    for (int pi = 0; pi < n_prompt && !g_stop; ++pi) {
        if ((long long) (pi + 1) * n_cont <= done) continue;
        const std::vector<int32_t> & pr = pre[pi];

        // prefix: llama.cpp master sequence and jspace keys/values
        llama_memory_clear(mem, true);
        for (int t0 = 0; t0 < P0; t0 += 512) {
            const int n = std::min<int>(512, (int) P0 - t0);
            for (int j = 0; j < n; ++j) {
                batch.token[j] = pr[t0 + j]; batch.pos[j] = t0 + j; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = n_par;
                batch.logits[j] = 0;
            }
            batch.n_tokens = n;
            if (llama_decode(ctx, batch) != 0) die("llama_decode failed (prefix)");
        }
        prefix_forward(pr);

        for (int bi = 0; bi < n_batches && !g_stop; ++bi) {
            const long long first = (long long) pi * n_cont + (long long) bi * n_par;
            if (first + n_par <= done) continue;

            // ---- generation: n_par continuations in parallel ----
            const double tg0 = now_sec();
            std::vector<std::vector<int32_t>> gen(n_par);
            std::vector<std::vector<float>> glp(n_par), gent(n_par);
            std::vector<std::mt19937_64> rng;
            for (int j = 0; j < n_par; ++j) {
                rng.emplace_back(seed * 0x9E3779B97F4A7C15ull + (uint64_t) pi * 1000003ull + (uint64_t) (bi * n_par + j));
                llama_memory_seq_rm(mem, j, -1, -1);
                llama_memory_seq_cp(mem, n_par, j, -1, -1);
                batch.token[j] = pr[P0]; batch.pos[j] = (int32_t) P0; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = j;
                batch.logits[j] = 1;
            }
            batch.n_tokens = n_par;
            for (int s = 0; s < G; ++s) {
                if (llama_decode(ctx, batch) != 0) die("llama_decode failed (gen)");
                std::vector<const float *> zr(n_par);
                for (int j = 0; j < n_par; ++j) zr[j] = llama_get_logits_ith(ctx, j);
                std::vector<double> uni(n_par);
                for (int j = 0; j < n_par; ++j) uni[j] = std::uniform_real_distribution<double>(0.0, 1.0)(rng[j]);
                parallel_for(n_par, std::min(n_par, n_threads), [&](int j) {
                    float * l = lp[j].data();
                    float m = -INFINITY;
                    for (int v = 0; v < n_vocab; ++v) m = std::max(m, zr[j][v]);
                    double se = 0.0;
                    for (int v = 0; v < n_vocab; ++v) se += std::exp((double) zr[j][v] - m);
                    const double lse = m + std::log(se);
                    double H = 0.0, tot = 0.0;
                    for (int v = 0; v < n_vocab; ++v) {
                        l[v] = (float) (zr[j][v] - lse);
                        const double pv = std::exp((double) l[v]);
                        H -= pv * l[v];
                        tot += pv;
                    }
                    const double target = uni[j] * tot;
                    double c = 0.0;
                    int tok = n_vocab - 1;
                    for (int v = 0; v < n_vocab; ++v) {
                        c += std::exp((double) l[v]);
                        if (c >= target) { tok = v; break; }
                    }
                    gen[j].push_back(tok);
                    glp[j].push_back(l[tok]);
                    gent[j].push_back((float) H);
                });
                if (s + 1 < G) {
                    for (int j = 0; j < n_par; ++j) {
                        batch.token[j] = gen[j].back(); batch.pos[j] = (int32_t) (P0 + 1 + s); batch.n_seq_id[j] = 1;
                        batch.seq_id[j][0] = j; batch.logits[j] = 1;
                    }
                    batch.n_tokens = n_par;
                }
            }
            t_gen += now_sec() - tg0;

            // ---- gradients, in batches of gb ----
            const double tb0 = now_sec();
            std::vector<float> Gall((size_t) n_par * nL * n_embd), Rall((size_t) n_par * nL * NW * n_embd);
            double nll_jspace = 0.0, ll_llama = 0.0;
            for (int g0 = 0; g0 < n_par; g0 += gb) {
                std::vector<std::vector<int32_t>> in(gb), nx(gb);
                for (int b = 0; b < gb; ++b) {
                    in[b].push_back(pr[P0]);
                    for (int k = 0; k + 1 < G; ++k) in[b].push_back(gen[g0 + b][k]);
                    nx[b] = gen[g0 + b];
                    for (float x : glp[g0 + b]) ll_llama += x;
                }
                std::vector<float> Gb, Rb;
                double nll = 0.0;
                grad_batch(in, nx, Gb, Rb, nll);
                nll_jspace += nll;
                std::copy(Gb.begin(), Gb.end(), Gall.begin() + (size_t) g0 * nL * n_embd);
                std::copy(Rb.begin(), Rb.end(), Rall.begin() + (size_t) g0 * nL * NW * n_embd);

                if (n_check > 0 && g0 < n_check && Lc < n_layer - 1) {
                    // repeat the same batch: the reused graphs must give the same G
                    std::vector<float> G2, R2;
                    double nll2 = 0.0;
                    grad_batch(in, nx, G2, R2, nll2);
                    double dmax = 0.0, gmax = 0.0;
                    for (size_t e = 0; e < Gb.size(); ++e) { dmax = std::max(dmax, (double) std::fabs(Gb[e] - G2[e])); gmax = std::max(gmax, (double) std::fabs(Gb[e])); }
                    fprintf(stderr, "check: recompute of the same batch: max |dG| %.3e (max |G| %.3e), loss %.6f vs %.6f\n", dmax, gmax, nll, nll2);
                    // jlens_value_exact on each whole text
                    for (int b = 0; b < gb && g0 + b < n_check; ++b) {
                        std::vector<int32_t> text(pr.begin(), pr.end());
                        text.insert(text.end(), gen[g0 + b].begin(), gen[g0 + b].end());
                        jlens_value_result res;
                        std::string err;
                        if (!jlens_value_exact(W, backend, buft, text, Lc, 32, true, res, err)) die(("value_exact: " + err).c_str());
                        if (g0 + b == 0) {
                            // how much does the reference itself move with its chunking?
                            for (int variant = 0; variant < 2; ++variant) {
                                jlens_value_result r2;
                                if (!jlens_value_exact(W, backend, buft, text, Lc, variant == 0 ? 64 : 32, true, r2, err, variant == 1))
                                    die(("value_exact: " + err).c_str());
                                double dh = 0, nh = 0, dg = 0, ng = 0;
                                for (int k = 0; k < G; ++k) {
                                    const size_t s = (size_t) (P0 + k);
                                    for (int e = 0; e < n_embd; ++e) {
                                        const double a1 = res.h[s * n_embd + e], a2 = r2.h[s * n_embd + e];
                                        dh += (a1 - a2) * (a1 - a2); nh += a1 * a1;
                                        const double g1 = (double) res.now[s * n_embd + e] + res.fut[s * n_embd + e];
                                        const double g2 = (double) r2.now[s * n_embd + e] + r2.fut[s * n_embd + e];
                                        dg += (g1 - g2) * (g1 - g2); ng += g1 * g1;
                                    }
                                }
                                fprintf(stderr, "check: reference chunk 32 vs %s: states rel diff %.3e, per-position gradients rel diff %.3e\n",
                                        variant == 0 ? "chunk 64" : "unchunked", std::sqrt(dh / nh), std::sqrt(dg / ng));
                                // this tool against the same variant: states per lag, and G
                                double mh = 0, mg = 0, ng2 = 0;
                                std::vector<double> g2(n_embd, 0.0);
                                std::string lagd;
                                for (int k = 0; k < G; ++k) {
                                    const size_t s = (size_t) (P0 + k);
                                    double dk = 0, nk = 0;
                                    for (int e = 0; e < n_embd; ++e) {
                                        const double a2 = r2.h[s * n_embd + e], a3 = hbufL[lc][(size_t) (b * G + k) * n_embd + e];
                                        dk += (a2 - a3) * (a2 - a3); nk += a2 * a2;
                                        g2[e] += (double) r2.now[s * n_embd + e] + r2.fut[s * n_embd + e];
                                    }
                                    mh += dk;
                                    if (k == 0 || k == 1 || k == 8 || k == 31) { char t[64]; snprintf(t, sizeof(t), " lag %d %.2e", k, std::sqrt(dk / nk)); lagd += t; }
                                }
                                for (int e = 0; e < n_embd; ++e) { const double x = Gb[((size_t) b * nL + lc) * n_embd + e]; mg += (x - g2[e]) * (x - g2[e]); ng2 += g2[e] * g2[e]; }
                                fprintf(stderr, "check: this tool vs %s: states rel diff %.3e (%s), G rel diff %.3e\n",
                                        variant == 0 ? "chunk 64" : "unchunked", std::sqrt(mh / nh), lagd.c_str(), std::sqrt(mg / ng2));
                            }
                        }
                        std::vector<double> gref(n_embd, 0.0);
                        double hd = 0.0, hn = 0.0;
                        for (int k = 0; k < G; ++k) {
                            const size_t s = (size_t) (P0 + k);
                            for (int e = 0; e < n_embd; ++e) {
                                gref[e] += (double) res.now[s * n_embd + e] + res.fut[s * n_embd + e];
                                const double dh = (double) res.h[s * n_embd + e] - hbufL[lc][(size_t) (b * G + k) * n_embd + e];
                                hd += dh * dh;
                                hn += (double) res.h[s * n_embd + e] * res.h[s * n_embd + e];
                            }
                        }
                        // hbuf holds the last computed batch (the repeat), the same continuations
                        double dot = 0.0, n1 = 0.0, n2 = 0.0, dd = 0.0;
                        for (int e = 0; e < n_embd; ++e) {
                            const double x = Gb[((size_t) b * nL + lc) * n_embd + e], y = gref[e];
                            dot += x * y; n1 += x * x; n2 += y * y; dd += (x - y) * (x - y);
                        }
                        double ll_b = 0.0;
                        for (float x : glp[g0 + b]) ll_b += x;
                        fprintf(stderr, "check: continuation %d: |G| %.4e, |G_ref| %.4e, cos %.6f, |G - G_ref| / |G_ref| %.3e; "
                                "readout states rel diff %.3e; log p(written) llama.cpp %.4f\n",
                                g0 + b, std::sqrt(n1), std::sqrt(n2), dot / std::sqrt(n1 * n2), std::sqrt(dd / n2), std::sqrt(hd / hn), ll_b);
                    }
                }
            }
            t_grad += now_sec() - tb0;

            // ---- write the records not yet in OUT.bin ----
            for (int j = 0; j < n_par; ++j) {
                if (first + j < done) continue;
                std::vector<int32_t> ri = { pi, bi * n_par + j };
                ri.insert(ri.end(), gen[j].begin(), gen[j].end());
                std::vector<float> rf;
                rf.insert(rf.end(), glp[j].begin(), glp[j].end());
                rf.insert(rf.end(), gent[j].begin(), gent[j].end());
                fwrite(ri.data(), sizeof(int32_t), ri.size(), fout);
                if (!multi) {
                    rf.insert(rf.end(), Gall.begin() + (size_t) j * n_embd, Gall.begin() + (size_t) (j + 1) * n_embd);
                    rf.insert(rf.end(), Rall.begin() + (size_t) j * NW * n_embd, Rall.begin() + (size_t) (j + 1) * NW * n_embd);
                    fwrite(rf.data(), sizeof(float), rf.size(), fout);
                } else {
                    fwrite(rf.data(), sizeof(float), rf.size(), fout);
                    std::vector<float> v32;
                    for (int li = 0; li < nL; ++li) {
                        const float * gsrc = Gall.data() + ((size_t) j * nL + li) * n_embd;
                        const float * rsrc = Rall.data() + ((size_t) j * nL + li) * NW * n_embd;
                        v32.insert(v32.end(), gsrc, gsrc + n_embd);
                        v32.insert(v32.end(), rsrc, rsrc + (size_t) NW * n_embd);
                    }
                    std::vector<ggml_fp16_t> v16(v32.size());
                    ggml_fp32_to_fp16_row(v32.data(), v16.data(), (int64_t) v32.size());
                    fwrite(v16.data(), sizeof(ggml_fp16_t), v16.size(), fout);
                }
                ++n_written;
            }
            fflush(fout);
            done = std::max(done, first + n_par);
            fprintf(stderr, "  prompt %3d batch %2d: log p(written) llama.cpp %.2f, jspace %.2f per continuation; gen %.1f s, grad %.1f s (total %lld/%lld)\n",
                    pi, bi, ll_llama / n_par, -nll_jspace / n_par, t_gen, t_grad, done, total);
            if (n_check > 0) { g_stop = 1; break; }

            {
                std::ifstream pf_ms(out_prefix + ".pause_ms");
                int v;
                if (pf_ms >> v) pause_ms = v;
            }
            for (int w = 0; w < pause_ms && !g_stop; w += 100) std::this_thread::sleep_for(std::chrono::milliseconds(std::min(100, pause_ms - w)));
        }
    }
    fclose(fout);
    if (g_stop && n_check == 0) fprintf(stderr, "homeo-score: stopped by signal; rerun to resume\n");
    fprintf(stderr, "homeo-score: %lld records written; gen %.1f s, grad %.1f s\n", n_written, t_gen, t_grad);

    ggml_gallocr_free(ga_pf);
    ggml_gallocr_free(ga_cf);
    ggml_gallocr_free(ga_uc);
    ggml_free(cf.gctx);
    for (auto & g : grp) ggml_free(g.uc.gctx);
    ggml_backend_buffer_free(st.buf);
    ggml_free(st.ctx);
    ggml_backend_free(backend);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
