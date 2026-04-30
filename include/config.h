#pragma once
# include <string>
# include <fstream>
# include "nlohmann/json.hpp"

struct modelConfig {
    int n_layer;
    int n_head;
    int n_embd;
    int vocab_size;
    int n_inner;
    int n_ctx;
    float layer_norm_eps;
    bool scale_attn_weights;
    int eos_token_id;
    int bos_token_id;

    static modelConfig load(std::string& path) {
        std::ifstream fin(path + "/config.json");
        nlohmann::json j = nlohmann::json::parse(fin);

        modelConfig cfg;
        cfg.n_layer = j["n_layer"].get<int>();
        cfg.n_head = j["n_head"].get<int>();
        cfg.n_embd = j["n_embd"].get<int>();
        cfg.vocab_size = j["vocab_size"].get<int>();
        cfg.n_inner = j["n_inner"].is_null() ? 4 * cfg.n_embd : j["n_inner"].get<int>();
        cfg.n_ctx = j["n_ctx"].get<int>();
        cfg.layer_norm_eps = j["layer_norm_epsilon"].get<float>();
        cfg.scale_attn_weights = j["scale_attn_weights"].get<bool>();
        cfg.eos_token_id = j["eos_token_id"].is_null() ? 50256 : j["eos_token_id"].get<int>();
        cfg.bos_token_id = j["bos_token_id"].is_null() ? 50256 : j["bos_token_id"].get<int>();

        return cfg;
    }

    int d_head() const { return n_embd / n_head; }
};
