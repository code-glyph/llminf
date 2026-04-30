#include "weights.h"
#include <torch/torch.h>
#include <fstream>
#include <string>

static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

GPT2Weights GPT2Weights::load(const std::string& path, const modelConfig& cfg) {
    torch::Device device(torch::kCUDA, 0);

    auto bytes = read_file(path + "/weights.pt");
    auto sd = torch::pickle_load(bytes).toGenericDict();

    GPT2Weights w;
    w.wte    = sd.at("transformer.wte.weight").toTensor().to(device);
    w.wpe    = sd.at("transformer.wpe.weight").toTensor().to(device);

    w.layers.resize(cfg.n_layer);
    for (int i = 0; i < cfg.n_layer; i++) {
        std::string p = "transformer.h." + std::to_string(i);
        auto& l = w.layers[i];
        l.ln_1_w   = sd.at(p + ".ln_1.weight").toTensor().to(device);
        l.ln_1_b   = sd.at(p + ".ln_1.bias").toTensor().to(device);
        l.c_attn_w = sd.at(p + ".attn.c_attn.weight").toTensor().to(device);
        l.c_attn_b = sd.at(p + ".attn.c_attn.bias").toTensor().to(device);
        l.c_proj_w = sd.at(p + ".attn.c_proj.weight").toTensor().to(device);
        l.c_proj_b = sd.at(p + ".attn.c_proj.bias").toTensor().to(device);
        l.ln_2_w   = sd.at(p + ".ln_2.weight").toTensor().to(device);
        l.ln_2_b   = sd.at(p + ".ln_2.bias").toTensor().to(device);
        l.c_fc_w   = sd.at(p + ".mlp.c_fc.weight").toTensor().to(device);
        l.c_fc_b   = sd.at(p + ".mlp.c_fc.bias").toTensor().to(device);
        l.c_fc2_w  = sd.at(p + ".mlp.c_proj.weight").toTensor().to(device);
        l.c_fc2_b  = sd.at(p + ".mlp.c_proj.bias").toTensor().to(device);
    }

    w.ln_f_w = sd.at("transformer.ln_f.weight").toTensor().to(device);
    w.ln_f_b = sd.at("transformer.ln_f.bias").toTensor().to(device);

    return w;
}