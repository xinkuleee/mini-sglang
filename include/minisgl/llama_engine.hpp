#pragma once
#include "minisgl/engine.hpp"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct msgl_handle;
namespace minisgl {
struct LlamaConfig {
    std::string model_path;
    std::uint32_t context_tokens = 4096;
    std::uint32_t batch_size = 512;
    std::uint32_t max_sequences = 64;
    std::int32_t threads = 4;
    std::int32_t gpu_layers = 0;
};

// Engine 的计算实现：RAII 管理 C ABI 所有权；调度与采样完全在 C++ core。
// 一个实例由 scheduler 所在线程独占，禁止复制以避免双重释放和 KV 并发修改。
class LlamaEngine final : public Engine {
  public:
    explicit LlamaEngine(const LlamaConfig &config);
    ~LlamaEngine() override;
    LlamaEngine(const LlamaEngine &) = delete;
    LlamaEngine &operator=(const LlamaEngine &) = delete;
    std::vector<Token> tokenize(const std::string &text) override;
    std::string piece(Token token) override;
    bool is_eog(Token token) const override;
    std::vector<std::vector<float>> forward(const std::vector<BatchToken> &batch) override;
    void copy_sequence(SequenceId src, SequenceId dst, std::size_t end_exclusive) override;
    void remove_sequence(SequenceId id) override;
    std::string
    apply_chat_template(const std::vector<std::pair<std::string, std::string>> &messages,
                        bool add_generation_prompt = true) const;

  private:
    msgl_handle *handle_ = nullptr;
    std::size_t vocab_size_ = 0;
};
} // namespace minisgl
