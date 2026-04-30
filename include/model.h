#pragma once
#include <torch/torch.h>
#include "config.h"
#include "weights.h"
#include "kvcache.h"

class GPT2Model {
public:
    GPT2Model(const modelConfig& cfg, const GPT2Weights& w)
        : cfg(cfg), w(w) {}

    // returns logits at last position: [vocab_size]
    torch::Tensor forward(torch::Tensor input_ids, KVCacheSimple& kv);

private:
    const modelConfig& cfg;
    const GPT2Weights& w;

    torch::Tensor layer_norm(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
    torch::Tensor attention(torch::Tensor x, int layer, KVCacheSimple& kv);
    torch::Tensor mlp(torch::Tensor x, int layer);
};