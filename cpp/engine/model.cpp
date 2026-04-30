    #include "model.h"
    #include <cmath>

    torch::Tensor GPT2Model::layer_norm(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
        return torch::layer_norm(x, {cfg.n_embd}, weight, bias, cfg.layer_norm_eps);
    }

    torch::Tensor GPT2Model::attention(torch::Tensor x, int layer, KVCacheSimple& kv) {
        auto& l = w.layers[layer];
        int seq_len = x.size(0);  

        auto qkv = torch::addmm(l.c_attn_b, x, l.c_attn_w);
        auto q = qkv.slice(1, 0,           cfg.n_embd);
        auto k = qkv.slice(1, cfg.n_embd,  2 * cfg.n_embd);
        auto v = qkv.slice(1, 2 * cfg.n_embd, 3 * cfg.n_embd);
        auto to_heads = [&](torch::Tensor t) {
            return t.view(std::vector<int64_t>{seq_len, cfg.n_head, cfg.d_head()})
                    .permute({1, 0, 2})
                    .contiguous();
        };
        q = to_heads(q);  // [n_head, seq, d_head]
        k = to_heads(k);
        v = to_heads(v);
        if (layer == 0) {
    std::cout << "qkv[-1,:3]: " << qkv[-1].slice(0,0,3) << "\n";
    std::cout << "q[-1,-1,:3]: " << q[-1][-1].slice(0,0,3) << "\n";
    std::cout << "k[-1,-1,:3]: " << k[-1][-1].slice(0,0,3) << "\n";
}

        kv.append(layer, k, v);

        auto K = kv.get_k(layer);
        auto V = kv.get_v(layer);

        float scale = cfg.scale_attn_weights ? 1.0f / std::sqrt(cfg.d_head()) : 1.0f;
        // std::cout << "q shape: " << q.sizes() << "\n";
        // std::cout << "K shape: " << K.sizes() << "\n";
        // std::cout << "K.T shape: " << K.transpose(1,2).sizes() << "\n";
        auto scores = torch::bmm(q, K.transpose(1, 2)) * scale;

        if (seq_len > 1) {
    // kv.past_len was already incremented by append() for layer 0;
    // total KV length == kv.past_len (it equals past_len_before + seq_len).
    int total_kv = kv.past_len;  // already includes current seq
    // mask[i,j] = -inf if j > (past_len_before + i), i.e. j is a future token
    int past_before = total_kv - seq_len;
    auto mask = torch::full({seq_len, total_kv}, 0.0f, x.options());
    for (int i = 0; i < seq_len; i++)
        for (int j = past_before + i + 1; j < total_kv; j++)
            mask[i][j] = -1e9f;
    scores = scores + mask.unsqueeze(0);
}

        auto attn = torch::softmax(scores, -1);

        auto ctx = torch::bmm(attn, V)
                    .permute({1, 0, 2})
                    .contiguous()
                    .view({seq_len, cfg.n_embd});

        if (layer == 0) {
            std::cout << "scores[-1,-1,:3]: " << scores[-1][-1].slice(0,0,3) << "\n";
            std::cout << "ctx[-1,:3]: " << ctx[-1].slice(0,0,3) << "\n";
        }

        return torch::addmm(l.c_proj_b, ctx, l.c_proj_w);
    }

    torch::Tensor GPT2Model::mlp(torch::Tensor x, int layer) {
        auto& l = w.layers[layer];

        auto h = torch::addmm(l.c_fc_b, x, l.c_fc_w);
        if (layer == 0) {
            std::cout << "mlp fc[-1,:3]: " << h[-1].slice(0,0,3) << "\n";
        }
        h = torch::gelu(h, "tanh");
        if (layer == 0) {
        std::cout << "mlp fc[-1,:3]: " << h[-1].slice(0,0,3) << "\n";
        // after gelu:
        std::cout << "mlp gelu[-1,:3]: " << h[-1].slice(0,0,3) << "\n";
        }
        auto out =torch::addmm(l.c_fc2_b, h, l.c_fc2_w);
        if (layer == 0) {
            std::cout << "mlp_out[-1,:3]: " << out[-1].slice(0,0,3) << "\n";
        }
        return out;
    }

    torch::Tensor GPT2Model::forward(torch::Tensor input_ids, KVCacheSimple& kv) {
        int seq_len = input_ids.size(0);

        auto tok_emb = w.wte.index({input_ids});                          // [seq, n_embd]
        auto pos_ids = torch::arange(kv.past_len, kv.past_len + seq_len,
                        input_ids.options());
        std::cout << "past_len: " << kv.past_len << "\n";
        std::cout << "pos_ids: " << pos_ids.sizes() << "\n";
        auto pos_emb = w.wpe.index({pos_ids});                            // [seq, n_embd]
        auto x = tok_emb + pos_emb;
        std::cout << "x[:3]: " << x.slice(0,0,1).slice(1,0,3) << "\n";

        // transformer layers
        for (int i = 0; i < cfg.n_layer; i++) {
            auto& l = w.layers[i];
            auto residual = x;
            x = layer_norm(x, l.ln_1_w, l.ln_1_b);
            auto attn_out = attention(x, i, kv);
            if (i == 0) std::cout << "attn_out[-1,:3]: " << attn_out[-1].slice(0,0,3) << "\n";
            x = residual + attn_out;
            residual = x;
            x = layer_norm(x, l.ln_2_w, l.ln_2_b);
            x = residual + mlp(x, i);
        }
        // kv.past_len += seq_len; 

        // final layer norm, return logits at last position only
        x = layer_norm(x, w.ln_f_w, w.ln_f_b);
        auto last = x[-1];                      // [n_embd]
        std::cout << "hidden[:3]: " << last.slice(0, 0, 3) << "\n";
        return torch::mm(last.unsqueeze(0), w.lm_head.t()).squeeze(0);
    }