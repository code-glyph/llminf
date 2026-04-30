#pragma once
#include <torch/torch.h>
#include "config.h"

struct KVCacheSimple {
    std::vector<torch::Tensor> cache;  // one tensor per layer
    int past_len = 0;

    KVCacheSimple(const modelConfig& cfg, int max_seq, torch::Device device) {
        cache.reserve(cfg.n_layer);
        for (int i = 0; i < cfg.n_layer; i++) {
            cache.push_back(torch::zeros(
                std::vector<int64_t>{2, cfg.n_head, max_seq, cfg.d_head()},
                torch::TensorOptions().dtype(torch::kFloat32).device(device)
            ));
        }
    }

    void append(int layer, torch::Tensor k, torch::Tensor v) {
    int seq_len = k.size(1);
    cache[layer].index_put_(
        {torch::tensor(0), torch::indexing::Slice(),
         torch::indexing::Slice(past_len, past_len + seq_len), torch::indexing::Slice()}, k);
    cache[layer].index_put_(
        {torch::tensor(1), torch::indexing::Slice(),
         torch::indexing::Slice(past_len, past_len + seq_len), torch::indexing::Slice()}, v);
    if (layer == 0) past_len += seq_len;  // only increment once, on first layer
}

torch::Tensor get_k(int layer) const {
    // cache[layer][0] shape: [n_head, max_seq, d_head]
    return cache[layer].index({0, torch::indexing::Slice(), 
                               torch::indexing::Slice(0, past_len), 
                               torch::indexing::Slice()});  // [n_head, past_len, d_head]
}
torch::Tensor get_v(int layer) const {
    return cache[layer].index({1, torch::indexing::Slice(),
                               torch::indexing::Slice(0, past_len),
                               torch::indexing::Slice()});  // [n_head, past_len, d_head]
}
};