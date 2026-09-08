#pragma once

// Engine 仅负责分词、模型前向和物理 KV 引用；调度、逻辑预算和采样留在
// C++ 核心。窄接口允许替换计算后端，生产实现不能用伪 logits 代替模型。
#include "minisgl/types.hpp"
#include <string>
#include <vector>

namespace minisgl {
struct BatchToken {
    Token token;
    std::size_t position;
    SequenceId sequence;
    bool logits;
};

class Engine {
  public:
    virtual ~Engine() = default;
    virtual std::vector<Token> tokenize(const std::string &text) = 0;
    virtual std::string piece(Token token) = 0;
    virtual bool is_eog(Token token) const = 0;
    // 只返回 logits=true 的行，顺序与 batch 中的行相同。
    virtual std::vector<std::vector<float>> forward(const std::vector<BatchToken> &batch) = 0;
    // 复制 [0,end_exclusive) 的 KV 引用；移除 src 后 dst 仍须有效。
    virtual void copy_sequence(SequenceId src, SequenceId dst, std::size_t end_exclusive) = 0;
    virtual void remove_sequence(SequenceId sequence) = 0;
};
} // namespace minisgl
