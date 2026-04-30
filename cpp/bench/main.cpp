#include <iostream>
#include "config.h"
#include "weights.h"
#include "model.h"
#include "generate.h"

int main() {
    std::string model_path = "../weights/gpt2-medium";
    torch::Device device(torch::kCUDA, 0);

    auto cfg = modelConfig::load(model_path);
    auto w   = GPT2Weights::load(model_path, cfg);
    GPT2Model model(cfg, w);

    std::vector<int> prompt = {464, 3290, 318, 257, 922};  // "The dog is a good"
    auto generated = greedy_generate(model, cfg, prompt, 50, device);

    std::cout << "\nGenerated token ids:\n";
    for (int id : generated) std::cout << id << " ";
    std::cout << "\n";

    // load ref inputs and run forward
    auto load_tensor = [](const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> bytes(std::istreambuf_iterator<char>(f), {});
    return torch::pickle_load(bytes).toTensor();
    };

    auto ref_input  = load_tensor("../weights/gpt2-medium/input_ids.pt").to(device).squeeze(0);
    auto ref_logits = load_tensor("../weights/gpt2-medium/logits.pt").to(device);
    KVCacheSimple kv2(cfg, cfg.n_ctx, device);
    auto cpp_logits = model.forward(ref_input, kv2);

    std::cout << "ref logits[:5]: ";
    for (int i = 0; i < 5; i++) std::cout << ref_logits[i].item<float>() << " ";
    std::cout << "\ncpp logits[:5]: ";
    for (int i = 0; i < 5; i++) std::cout << cpp_logits[i].item<float>() << " ";

    float max_diff = (cpp_logits - ref_logits).abs().max().item<float>();
    std::cout << "\nmax diff: " << max_diff << "\n";
    std::cout << (max_diff < 1e-2 ? "PASS" : "FAIL") << "\n";
    return 0;
}