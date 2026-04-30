#include <iostream>
#include "config.h"

int main() {
    std::string path = "../weights/gpt2-medium";

    auto cfg = modelConfig::load(path);
    std::cout << "n_layer: "    << cfg.n_layer    << "\n";
    std::cout << "n_head: "     << cfg.n_head     << "\n";
    std::cout << "n_embd: "     << cfg.n_embd     << "\n";
    std::cout << "d_head: "     << cfg.d_head()   << "\n";  
    std::cout << "vocab_size: " << cfg.vocab_size  << "\n";
    std::cout << "n_inner: " << cfg.n_inner << "\n";
    std::cout << "n_ctx: " << cfg.n_ctx << "\n";
    std::cout << "layer_norm_eps: " << cfg.layer_norm_eps << "\n";
    std::cout << "scale_attn_weights: " << cfg.scale_attn_weights << "\n";
    std::cout << "eos_token_id: " << cfg.eos_token_id << "\n";
    std::cout << "bos_token_id: " << cfg.bos_token_id << "\n";
    return 0;
}