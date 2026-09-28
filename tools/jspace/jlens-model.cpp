#include "jlens.h"

#include "ggml-alloc.h"
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

ggml_tensor * jlens_model::get(const std::string & name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        fprintf(stderr, "%s: tensor not found: %s\n", __func__, name.c_str());
        return nullptr;
    }
    return it->second;
}

ggml_tensor * jlens_model::blk(const char * fmt, int il) const {
    char name[128];
    snprintf(name, sizeof(name), fmt, il);
    return get(name);
}

static bool gguf_get_i64(const gguf_context * gguf, const char * key, int64_t & v) {
    const int kid = gguf_find_key(gguf, key);
    if (kid < 0) return false;
    v = (int64_t) gguf_get_val_u32(gguf, kid);
    return true;
}

bool jlens_model_load(const char * path, ggml_backend_buffer_type_t buft, jlens_model & out) {
    // map the whole file
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return false; }
    struct stat st; fstat(fd, &st);
    out.file_size = st.st_size;
    out.file_map  = mmap(nullptr, out.file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (out.file_map == MAP_FAILED) { perror("mmap"); return false; }

    struct gguf_init_params gparams = { /*no_alloc=*/true, /*ctx=*/nullptr };
    out.gguf = gguf_init_from_file(path, gparams);
    if (!out.gguf) { fprintf(stderr, "%s: gguf_init_from_file failed\n", __func__); return false; }

    // hparams
    auto & hp = out.hparams;
    bool ok = true;
    ok &= gguf_get_i64(out.gguf, "glm4.embedding_length",            hp.n_embd);
    ok &= gguf_get_i64(out.gguf, "glm4.block_count",                 hp.n_layer);
    ok &= gguf_get_i64(out.gguf, "glm4.attention.head_count",        hp.n_head);
    ok &= gguf_get_i64(out.gguf, "glm4.attention.head_count_kv",     hp.n_head_kv);
    ok &= gguf_get_i64(out.gguf, "glm4.feed_forward_length",         hp.n_ff);
    ok &= gguf_get_i64(out.gguf, "glm4.rope.dimension_count",        hp.n_rot);
    ok &= gguf_get_i64(out.gguf, "glm4.context_length",              hp.n_ctx_train);
    if (!ok) { fprintf(stderr, "%s: missing glm4 hparams in GGUF metadata\n", __func__); return false; }
    {
        const int kid = gguf_find_key(out.gguf, "glm4.rope.freq_base");
        if (kid >= 0) hp.rope_freq_base = gguf_get_val_f32(out.gguf, kid);
        const int eid = gguf_find_key(out.gguf, "glm4.attention.layer_norm_rms_epsilon");
        if (eid >= 0) hp.norm_eps = gguf_get_val_f32(out.gguf, eid);
    }
    hp.n_embd_head = hp.n_embd / hp.n_head;

    // vocab tokens (for readout)
    {
        const int kid = gguf_find_key(out.gguf, "tokenizer.ggml.tokens");
        if (kid >= 0 && gguf_get_kv_type(out.gguf, kid) == GGUF_TYPE_ARRAY) {
            const int64_t n = gguf_get_arr_n(out.gguf, kid);
            out.vocab_tokens.reserve(n);
            for (int64_t i = 0; i < n; ++i) out.vocab_tokens.push_back(gguf_get_arr_str(out.gguf, kid, i));
        }
    }
    hp.n_vocab = (int64_t) out.vocab_tokens.size();

    // create weight tensors (no data), then allocate one backend buffer and upload
    const int64_t n_tensors = gguf_get_n_tensors(out.gguf);
    struct ggml_init_params iparams = {
        /*mem_size  =*/ (size_t) n_tensors * ggml_tensor_overhead(),
        /*mem_buffer=*/ nullptr,
        /*no_alloc  =*/ true,
    };
    out.wctx = ggml_init(iparams);

    const size_t data_base = gguf_get_data_offset(out.gguf);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char *        name = gguf_get_tensor_name(out.gguf, i);
        const int64_t *     ne   = gguf_get_tensor_ne(out.gguf, i);
        const enum ggml_type type = gguf_get_tensor_type(out.gguf, i);

        int n_dims = 0;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) if (ne[d] != 1 || d == 0) n_dims = d + 1;

        ggml_tensor * t = ggml_new_tensor(out.wctx, type, n_dims, ne);
        ggml_set_name(t, name);
        out.tensors[name] = t;
    }

    out.wbuf = ggml_backend_alloc_ctx_tensors_from_buft(out.wctx, buft);
    if (!out.wbuf) { fprintf(stderr, "%s: failed to allocate weight buffer (%.2f GiB needed?)\n", __func__,
                       (double) out.file_size / (1 << 30)); return false; }

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name   = gguf_get_tensor_name(out.gguf, i);
        const size_t offset = data_base + gguf_get_tensor_offset(out.gguf, i);
        const size_t size   = gguf_get_tensor_size(out.gguf, i);
        ggml_tensor * t = out.tensors[name];
        ggml_backend_tensor_set(t, (const uint8_t *) out.file_map + offset, 0, size);
    }

    fprintf(stderr, "%s: loaded %lld tensors (%.2f GiB), n_layer=%lld n_embd=%lld n_head=%lld/%lld n_ff=%lld n_vocab=%lld\n",
            __func__, (long long) n_tensors, (double) out.file_size / (1 << 30),
            (long long) hp.n_layer, (long long) hp.n_embd, (long long) hp.n_head,
            (long long) hp.n_head_kv, (long long) hp.n_ff, (long long) hp.n_vocab);
    return true;
}

void jlens_model_free(jlens_model & m) {
    if (m.wbuf) ggml_backend_buffer_free(m.wbuf);
    if (m.wctx) ggml_free(m.wctx);
    if (m.gguf) gguf_free(m.gguf);
    if (m.file_map && m.file_map != MAP_FAILED) munmap(m.file_map, m.file_size);
    m = jlens_model{};
}
