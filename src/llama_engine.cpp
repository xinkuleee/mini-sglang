#include "minisgl/llama_engine.hpp"
#include "minisgl_bridge.h"
#include <limits>
#include <stdexcept>

namespace minisgl {
namespace {
void checked(int status) {
    if (status < 0)
        throw std::runtime_error(msgl_error());
}
std::int32_t checked_position(std::size_t value) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::overflow_error("position exceeds llama int32 range");
    return static_cast<std::int32_t>(value);
}
void no_nul(const std::string &text) {
    if (text.find('\0') != std::string::npos)
        throw std::invalid_argument("chat fields and model path cannot contain NUL");
}
} // namespace
LlamaEngine::LlamaEngine(const LlamaConfig &config) {
    no_nul(config.model_path);
    msgl_config native{config.model_path.c_str(), config.context_tokens, config.batch_size,
                       config.max_sequences,      config.threads,        config.gpu_layers};
    handle_ = msgl_create(&native);
    if (!handle_)
        throw std::runtime_error(msgl_error());
    const auto vocab = msgl_vocab_size(handle_);
    if (vocab <= 0) {
        const std::string error = msgl_error();
        msgl_destroy(handle_);
        handle_ = nullptr;
        throw std::runtime_error(error);
    }
    vocab_size_ = static_cast<std::size_t>(vocab);
}
LlamaEngine::~LlamaEngine() { msgl_destroy(handle_); }
std::vector<Token> LlamaEngine::tokenize(const std::string &text) {
    std::size_t required = 0;
    checked(msgl_tokenize(handle_, text.data(), text.size(), nullptr, 0, &required));
    std::vector<Token> output(required);
    const int status =
        msgl_tokenize(handle_, text.data(), text.size(), output.data(), output.size(), &required);
    checked(status);
    if (status != 0)
        throw std::runtime_error("tokenization size changed");
    output.resize(required);
    return output;
}
std::string LlamaEngine::piece(Token value) {
    std::size_t required = 0;
    checked(msgl_piece(handle_, value, nullptr, 0, &required));
    std::string output(required, '\0');
    const int status = msgl_piece(handle_, value, output.data(), output.size(), &required);
    checked(status);
    if (status != 0)
        throw std::runtime_error("piece size changed");
    output.resize(required);
    return output;
}
bool LlamaEngine::is_eog(Token value) const {
    const int status = msgl_is_eog(handle_, value);
    checked(status);
    return status != 0;
}
std::vector<std::vector<float>> LlamaEngine::forward(std::span<const BatchToken> batch) {
    std::vector<msgl_batch_token> native;
    native.reserve(batch.size());
    std::size_t rows = 0;
    for (const auto &item : batch) {
        native.push_back(
            {item.token, checked_position(item.position), item.sequence, item.logits ? 1 : 0});
        rows += item.logits;
    }
    if (rows > std::numeric_limits<std::size_t>::max() / vocab_size_)
        throw std::overflow_error("logits buffer size overflow");
    std::vector<float> flat(rows * vocab_size_);
    checked(msgl_forward(handle_, native.data(), native.size(), flat.data(), flat.size()));
    std::vector<std::vector<float>> output;
    output.reserve(rows);
    for (std::size_t i = 0; i < rows; ++i)
        output.emplace_back(flat.begin() + i * vocab_size_, flat.begin() + (i + 1) * vocab_size_);
    return output;
}
void LlamaEngine::copy_sequence(SequenceId src, SequenceId dst, std::size_t end) {
    checked(msgl_copy_seq(handle_, src, dst, checked_position(end)));
}
void LlamaEngine::remove_sequence(SequenceId seq) { checked(msgl_remove_seq(handle_, seq)); }
std::string
LlamaEngine::apply_chat_template(std::span<const std::pair<std::string, std::string>> messages,
                                 bool add_generation_prompt) const {
    std::vector<msgl_chat_message> native;
    native.reserve(messages.size());
    for (const auto &message : messages) {
        no_nul(message.first);
        no_nul(message.second);
        native.push_back({message.first.c_str(), message.second.c_str()});
    }
    std::size_t required = 0;
    checked(msgl_apply_chat_template(handle_, native.data(), native.size(), add_generation_prompt,
                                     nullptr, 0, &required));
    std::string output(required, '\0');
    const int status =
        msgl_apply_chat_template(handle_, native.data(), native.size(), add_generation_prompt,
                                 output.data(), output.size(), &required);
    checked(status);
    if (status != 0)
        throw std::runtime_error("chat template size changed");
    output.resize(required);
    return output;
}
} // namespace minisgl
