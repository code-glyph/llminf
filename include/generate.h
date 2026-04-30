#pragma once
#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include "model.h"
#include "kvcache.h"

std::vector<int> greedy_generate(
    GPT2Model& model,
    const modelConfig& cfg,
    std::vector<int> prompt_ids,
    int max_new_tokens,
    torch::Device device
);