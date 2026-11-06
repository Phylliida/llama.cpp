// dflash-dump: training data for distilling a DFlash draft against this exact target (K3 research, TEMP tool)
//
// Runs the target over documents (prompt processing only) and writes, per token:
//   - the input embeddings of the draft's target layers (what the drafter's fc reads at inference), bf16
//   - the target's top-K next-token log-probabilities (temperature 1)
//   - the token id and a train flag (assistant spans)
//
// Documents (DUMP_DOCS) use a length-prefixed text format written by k3-research/distill/docs_to_seg.py:
//   "D <n_segments>\n" then per segment "S <train 0|1> <n_bytes>\n<bytes>\n"
// Each segment is tokenized with special-token parsing; documents are truncated at DUMP_MAX_DOC tokens.
//
// env: DUMP_DOCS=<file> DUMP_OUT=<dir> DUMP_TAPS=20,38,55,67,79,91 DUMP_TOPK=32 DUMP_MAX_DOC=4096 DUMP_MAX_TOKENS=<total>
// args: the usual model / offload arguments (-m, -c, -b/-ub, -ngl, -ncmoe, -ts, ...)
// output (appended): tokens.i32, mask.u8, taps.bf16 [n, n_taps, n_embd], topk_id.i32 [n, K], topk_lp.f32 [n, K],
//                    docs.idx (one "<first token> <n_tokens>" line per document)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static std::string env_str(const char * name, const char * def) {
    const char * v = getenv(name);
    return v ? v : def;
}

struct segment {
    std::string text;
    bool train;
};

static bool read_doc(std::ifstream & in, std::vector<segment> & segs) {
    segs.clear();
    std::string line;
    if (!std::getline(in, line)) {
        return false;
    }
    int n = 0;
    if (sscanf(line.c_str(), "D %d", &n) != 1) {
        return false;
    }
    for (int i = 0; i < n; ++i) {
        if (!std::getline(in, line)) {
            return false;
        }
        int train = 0;
        long nb = 0;
        if (sscanf(line.c_str(), "S %d %ld", &train, &nb) != 2) {
            return false;
        }
        std::string text((size_t) nb, '\0');
        in.read(text.data(), nb);
        in.get(); // newline after the bytes
        segs.push_back({ std::move(text), train != 0 });
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx = 4096;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    const std::string docs_path = env_str("DUMP_DOCS", "");
    const std::string out_dir   = env_str("DUMP_OUT", "");
    const int topk      = atoi(env_str("DUMP_TOPK", "32").c_str());
    const int max_doc   = atoi(env_str("DUMP_MAX_DOC", "4096").c_str());
    const long max_tok  = atol(env_str("DUMP_MAX_TOKENS", "1000000000").c_str());
    std::vector<uint32_t> taps;
    {
        std::stringstream ss(env_str("DUMP_TAPS", "20,38,55,67,79,91"));
        std::string t;
        while (std::getline(ss, t, ',')) {
            taps.push_back((uint32_t) atoi(t.c_str()));
        }
    }
    if (docs_path.empty() || out_dir.empty()) {
        LOG_ERR("set DUMP_DOCS and DUMP_OUT\n");
        return 1;
    }
    params.n_ctx = std::max(params.n_ctx, max_doc);

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to load the model / create the context\n");
        return 1;
    }
    for (uint32_t lid : taps) {
        llama_set_embeddings_layer_inp(ctx, lid, true);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);
    const int n_ub    = (int) llama_n_ubatch(ctx);

    auto open_out = [&](const char * name) {
        FILE * f = fopen((out_dir + "/" + name).c_str(), "ab");
        if (!f) {
            LOG_ERR("cannot open %s/%s\n", out_dir.c_str(), name);
            exit(1);
        }
        return f;
    };
    FILE * f_tok  = open_out("tokens.i32");
    FILE * f_mask = open_out("mask.u8");
    FILE * f_taps = open_out("taps.bf16");
    FILE * f_kid  = open_out("topk_id.i32");
    FILE * f_klp  = open_out("topk_lp.f32");
    FILE * f_idx  = open_out("docs.idx");

    // continue the token count of earlier runs (append mode)
    long n_written = 0;
    {
        fseek(f_tok, 0, SEEK_END);
        n_written = ftell(f_tok) / (long) sizeof(int32_t);
    }

    std::ifstream in(docs_path, std::ios::binary);
    std::vector<segment> segs;
    llama_batch batch = llama_batch_init(n_ub, 0, 1);
    std::vector<ggml_bf16_t> taps_buf;
    std::vector<int32_t> kid_buf;
    std::vector<float>   klp_buf;

    const auto t0 = std::chrono::steady_clock::now();
    long n_done = 0;
    int  n_docs = 0;
    const int n_threads_post = (int) std::max(1u, std::thread::hardware_concurrency() / 2);

    while (n_done < max_tok && read_doc(in, segs)) {
        std::vector<llama_token> toks;
        std::vector<uint8_t>     mask;
        for (const auto & s : segs) {
            std::vector<llama_token> t = common_tokenize(vocab, s.text, false, true);
            toks.insert(toks.end(), t.begin(), t.end());
            mask.insert(mask.end(), t.size(), s.train ? 1 : 0);
        }
        if ((int) toks.size() > max_doc) {
            toks.resize(max_doc);
            mask.resize(max_doc);
        }
        if (toks.size() < 16) {
            continue;
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        fprintf(f_idx, "%ld %zu\n", n_written, toks.size());

        for (size_t beg = 0; beg < toks.size(); beg += n_ub) {
            const int n = (int) std::min<size_t>(n_ub, toks.size() - beg);
            common_batch_clear(batch);
            for (int i = 0; i < n; ++i) {
                common_batch_add(batch, toks[beg + i], (llama_pos) (beg + i), { 0 }, true);
            }
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("llama_decode failed\n");
                return 1;
            }

            // taps: [n, n_taps, n_embd] bf16
            taps_buf.resize((size_t) n * taps.size() * n_embd);
            for (size_t k = 0; k < taps.size(); ++k) {
                const float * src = llama_get_embeddings_layer_inp(ctx, taps[k]);
                if (!src) {
                    LOG_ERR("tap %u not extracted\n", taps[k]);
                    return 1;
                }
                for (int i = 0; i < n; ++i) {
                    ggml_fp32_to_bf16_row(src + (size_t) i * n_embd,
                            taps_buf.data() + ((size_t) i * taps.size() + k) * n_embd, n_embd);
                }
            }

            // top-K log-probs of every position, in parallel
            kid_buf.resize((size_t) n * topk);
            klp_buf.resize((size_t) n * topk);
            auto work = [&](int r0, int r1) {
                std::vector<int32_t> idx(n_vocab);
                for (int r = r0; r < r1; ++r) {
                    const float * lg = llama_get_logits_ith(ctx, r);
                    float mx = -INFINITY;
                    for (int v = 0; v < n_vocab; ++v) {
                        mx = std::max(mx, lg[v]);
                    }
                    double se = 0.0;
                    for (int v = 0; v < n_vocab; ++v) {
                        se += std::exp((double) (lg[v] - mx));
                    }
                    const float lse = mx + (float) std::log(se);
                    std::iota(idx.begin(), idx.end(), 0);
                    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                            [&](int32_t a, int32_t b) { return lg[a] > lg[b]; });
                    for (int j = 0; j < topk; ++j) {
                        kid_buf[(size_t) r * topk + j] = idx[j];
                        klp_buf[(size_t) r * topk + j] = lg[idx[j]] - lse;
                    }
                }
            };
            std::vector<std::thread> th;
            const int per = (n + n_threads_post - 1) / n_threads_post;
            for (int t = 0; t < n_threads_post; ++t) {
                const int r0 = t * per, r1 = std::min(n, r0 + per);
                if (r0 < r1) {
                    th.emplace_back(work, r0, r1);
                }
            }
            for (auto & t : th) {
                t.join();
            }

            fwrite(toks.data() + beg, sizeof(int32_t), n, f_tok);
            fwrite(mask.data() + beg, 1, n, f_mask);
            fwrite(taps_buf.data(), sizeof(ggml_bf16_t), taps_buf.size(), f_taps);
            fwrite(kid_buf.data(), sizeof(int32_t), kid_buf.size(), f_kid);
            fwrite(klp_buf.data(), sizeof(float), klp_buf.size(), f_klp);
            n_written += n;
            n_done += n;
        }
        for (FILE * f : { f_tok, f_mask, f_taps, f_kid, f_klp, f_idx }) {
            fflush(f);
        }
        ++n_docs;
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        LOG_INF("doc %d: %zu tokens; total %ld tokens in %.0f s (%.1f tok/s)\n", n_docs, toks.size(), n_done, dt, n_done / dt);
    }

    for (FILE * f : { f_tok, f_mask, f_taps, f_kid, f_klp, f_idx }) {
        fclose(f);
    }
    llama_batch_free(batch);
    LOG_INF("done: %d documents, %ld tokens\n", n_docs, n_done);
    return 0;
}
