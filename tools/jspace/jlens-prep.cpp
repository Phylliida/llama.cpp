// jlens-prep: tokenize a UTF-8 text file into a raw little-endian i32 token-id
// stream for jlens-fit --tokens-file, using the model's own GGUF tokenizer
// (vocab-only load, no weights).
//
// usage: jlens-prep -m model.gguf --text corpus.txt --out ids.bin
//        [--max-tokens N] [--add-bos]

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void die(const char * msg) { fprintf(stderr, "jlens-prep: %s\n", msg); exit(1); }

int main(int argc, char ** argv) {
    std::string model_path, text_path, out_path;
    int64_t max_tokens = 0;
    bool add_bos = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "-m")           model_path = argv[++i];
        else if (a == "--text")       text_path  = argv[++i];
        else if (a == "--out")        out_path   = argv[++i];
        else if (a == "--max-tokens") max_tokens = atoll(argv[++i]);
        else if (a == "--add-bos")    add_bos    = true;
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || text_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: jlens-prep -m model.gguf --text corpus.txt --out ids.bin "
                        "[--max-tokens N] [--add-bos]\n");
        return 1;
    }

    FILE * f = fopen(text_path.c_str(), "rb");
    if (!f) die("cannot open text file");
    fseek(f, 0, SEEK_END);
    const long nbytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (nbytes <= 0) { fclose(f); die("empty text file"); }
    std::string text((size_t) nbytes, '\0');
    if (fread(text.data(), 1, text.size(), f) != text.size()) { fclose(f); die("short read"); }
    fclose(f);

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) die("model (vocab) load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_special = add_bos;
    const int32_t rv = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                                      nullptr, 0, add_special, /*parse_special=*/false);
    if (rv >= 0) die("unexpected tokenize probe result");
    std::vector<int32_t> ids((size_t) -rv);
    const int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                                     ids.data(), (int32_t) ids.size(), add_special, false);
    if (n < 0) die("tokenize failed");
    ids.resize((size_t) n);
    if (max_tokens > 0 && (int64_t) ids.size() > max_tokens) ids.resize((size_t) max_tokens);

    f = fopen(out_path.c_str(), "wb");
    if (!f) die("cannot open output");
    if (fwrite(ids.data(), sizeof(int32_t), ids.size(), f) != ids.size()) { fclose(f); die("write failed"); }
    fclose(f);

    fprintf(stderr, "jlens-prep: %zu tokens (%ld text bytes) -> %s\n", ids.size(), nbytes, out_path.c_str());

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
