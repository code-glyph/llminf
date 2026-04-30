#include "generate.h"

std::vector<int> greedy_generate(
    GPT2Model& model,
    const modelConfig& cfg,
    std::vector<int> prompt_ids,
    int max_new_tokens,
    torch::Device device
) {
    KVCacheSimple kv(cfg, cfg.n_ctx, device);

    // prefill — process all prompt tokens at once
    auto input = torch::tensor(prompt_ids, 
                     torch::TensorOptions().dtype(torch::kLong).device(device));
    auto logits = model.forward(input, kv);  // [vocab_size]

    // greedy sample first token
    int next_id = logits.argmax().item<int>();
    prompt_ids.push_back(next_id);

    // decode loop
    std::vector<double> step_times;

    for (int step = 0; step < max_new_tokens - 1; step++) {
        if (next_id == cfg.eos_token_id) break;

        auto t0 = std::chrono::high_resolution_clock::now();

        input = torch::tensor({next_id},
                    torch::TensorOptions().dtype(torch::kLong).device(device));
        logits = model.forward(input, kv);
        next_id = logits.argmax().item<int>();
        prompt_ids.push_back(next_id);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        step_times.push_back(ms);
    }

    // report timing
    if (!step_times.empty()) {
        std::sort(step_times.begin(), step_times.end());
        int n = step_times.size();
        double p50 = step_times[n * 0.50];
        double p95 = step_times[n * 0.95];
        double mean = std::accumulate(step_times.begin(), step_times.end(), 0.0) / n;
        std::cout << "\n--- decode timing (" << n << " steps) ---\n";
        std::cout << "mean:  " << mean << " ms/tok\n";
        std::cout << "p50:   " << p50  << " ms/tok\n";
        std::cout << "p95:   " << p95  << " ms/tok\n";
    }

    return prompt_ids;
}