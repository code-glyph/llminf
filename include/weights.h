#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include "config.h"

struct LayerWeights {
    // attention
    torch::Tensor ln_1_w, ln_1_b;
    torch::Tensor c_attn_w, c_attn_b;   // fused QKV [n_embd, 3*n_embd]
    torch::Tensor c_proj_w, c_proj_b;
    // mlp
    torch::Tensor ln_2_w, ln_2_b;
    torch::Tensor c_fc_w, c_fc_b;
    torch::Tensor c_fc2_w, c_fc2_b;
};

struct GPT2Weights {
    torch::Tensor wte;   // [vocab_size, n_embd]
    torch::Tensor wpe;   // [n_ctx, n_embd]
    std::vector<LayerWeights> layers;
    torch::Tensor ln_f_w, ln_f_b;

    static GPT2Weights load(const std::string& path, const modelConfig& cfg);
};