#include "minisgl_bridge.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local char last_error[2048] = {};
void record_error(const char *message) noexcept {
    std::snprintf(last_error, sizeof(last_error), "%s", message ? message : "unknown native error");
}
template <class F> int32_t protect(F &&function) noexcept {
    last_error[0] = '\0';
    try {
        return function();
    } catch (const std::exception &e) {
        record_error(e.what());
    } catch (...) {
        record_error("unknown C++ exception in llama adapter");
    }
    return -1;
}
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
int32_t checked_i32(size_t value) {
    require(value <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
            "value exceeds llama int32 range");
    return static_cast<int32_t>(value);
}
/* llama 的 backend 注册表是进程全局对象；按进程初始化，避免一个 Engine
 * 的析构释放另一个 Engine 仍在使用的 backend。模型和 context 则逐实例释放。 */
void initialize_backend() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_backend_init();
        ggml_backend_load_all();
    });
}
} // namespace

struct msgl_handle {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
    const llama_vocab *vocab = nullptr;
    uint32_t batch_size = 0;
    uint32_t max_sequences = 0;
    uint32_t context_tokens = 0;
    int32_t vocab_size = 0;
};
namespace {
void valid(msgl_handle *h) { require(h && h->context && h->vocab, "invalid engine handle"); }
void sequence(msgl_handle *h, int32_t seq) {
    require(seq >= 0 && static_cast<uint32_t>(seq) < h->max_sequences,
            "sequence id outside configured capacity");
}
void token(msgl_handle *h, int32_t value) {
    require(value >= 0 && value < h->vocab_size, "token id outside vocabulary");
}
} // namespace

extern "C" {
const char *msgl_error() noexcept { return last_error; }
msgl_handle *msgl_create(const msgl_config *config) noexcept {
    msgl_handle *result = nullptr;
    protect([&]() -> int32_t {
        require(config && config->model_path && config->model_path[0], "model_path is required");
        require(config->context_tokens > 0 && config->batch_size > 0 && config->max_sequences > 0,
                "context, batch and sequence capacities must be positive");
        checked_i32(config->context_tokens);
        checked_i32(config->batch_size);
        checked_i32(config->max_sequences);
        require(config->threads > 0 && config->gpu_layers >= 0,
                "threads must be positive and gpu_layers nonnegative");
        initialize_backend();
        auto h = std::make_unique<msgl_handle>();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = config->gpu_layers;
        h->model.reset(llama_model_load_from_file(config->model_path, mp));
        require(h->model != nullptr, "cannot load GGUF model (see llama diagnostics)");
        require(!llama_model_is_recurrent(h->model.get()) && !llama_model_is_hybrid(h->model.get()),
                "recurrent/hybrid models are unsupported: prefix KV references require ordinary "
                "transformer attention");
        require(!llama_model_has_encoder(h->model.get()) &&
                    llama_model_has_decoder(h->model.get()) &&
                    !llama_model_is_diffusion(h->model.get()),
                "only causal decoder-only transformer models are supported");
        h->vocab = llama_model_get_vocab(h->model.get());
        require(h->vocab != nullptr, "model has no vocabulary");
        h->vocab_size = llama_vocab_n_tokens(h->vocab);
        require(h->vocab_size > 0, "model vocabulary is empty");
        auto cp = llama_context_default_params();
        cp.n_ctx = config->context_tokens;
        cp.n_batch = config->batch_size;
        cp.n_ubatch = config->batch_size;
        cp.n_seq_max = config->max_sequences;
        cp.n_threads = config->threads;
        cp.n_threads_batch = config->threads;
        cp.n_outputs_max_per_seq = 0; // Engine 契约允许同一序列请求多个 logits 行。
        cp.kv_unified = true;         // 总 token 池；不同请求的共同前缀引用同一 KV cell。
        cp.swa_full = true;           // 不自动遗弃 sliding-window 前缀，以维持 radix 引用生命周期。
        cp.no_perf = true;
        h->context.reset(llama_init_from_model(h->model.get(), cp));
        require(h->context != nullptr, "cannot allocate llama context (see llama diagnostics)");
        require(llama_get_memory(h->context.get()) != nullptr, "model has no shareable KV memory");
        require(llama_n_ctx(h->context.get()) >= config->context_tokens &&
                    llama_n_batch(h->context.get()) >= config->batch_size &&
                    llama_n_seq_max(h->context.get()) >= config->max_sequences,
                "llama allocated less capacity than requested");
        h->context_tokens = config->context_tokens;
        h->batch_size = config->batch_size;
        h->max_sequences = config->max_sequences;
        result = h.release();
        return 0;
    });
    return result;
}
void msgl_destroy(msgl_handle *h) noexcept {
    // unique_ptr 顺序使 context 先于 model 析构。C ABI 永远不传播异常。
    try {
        delete h;
    } catch (...) {
        record_error("exception while freeing llama engine");
    }
}
int32_t msgl_vocab_size(msgl_handle *h) noexcept {
    return protect([&] {
        valid(h);
        return h->vocab_size;
    });
}
int32_t msgl_tokenize(msgl_handle *h, const char *text, size_t length, int32_t *out,
                      size_t capacity, size_t *required) noexcept {
    return protect([&]() -> int32_t {
        valid(h);
        require(required && (text || !length) && (out || !capacity), "invalid tokenize buffer");
        const int32_t n = llama_tokenize(h->vocab, text ? text : "", checked_i32(length), out,
                                         checked_i32(capacity), true, true);
        require(n != std::numeric_limits<int32_t>::min(), "tokenization exceeds int32 range");
        *required = static_cast<size_t>(n < 0 ? -n : n);
        return n < 0 ? 1 : 0;
    });
}
int32_t msgl_piece(msgl_handle *h, int32_t value, char *out, size_t capacity,
                   size_t *required) noexcept {
    return protect([&]() -> int32_t {
        valid(h);
        token(h, value);
        require(required && (out || !capacity), "invalid piece buffer");
        const int32_t n =
            llama_token_to_piece(h->vocab, value, out, checked_i32(capacity), 0, false);
        require(n != std::numeric_limits<int32_t>::min(), "token piece exceeds int32 range");
        *required = static_cast<size_t>(n < 0 ? -n : n);
        return n < 0 ? 1 : 0;
    });
}
int32_t msgl_is_eog(msgl_handle *h, int32_t value) noexcept {
    return protect([&]() -> int32_t {
        valid(h);
        token(h, value);
        return llama_vocab_is_eog(h->vocab, value) ? 1 : 0;
    });
}
int32_t msgl_forward(msgl_handle *h, const msgl_batch_token *input, size_t count, float *output,
                     size_t output_count) noexcept {
    return protect([&]() -> int32_t {
        valid(h);
        require(input || !count, "null batch");
        require(count <= h->batch_size, "batch exceeds configured capacity");
        size_t rows = 0;
        for (size_t i = 0; i < count; ++i) {
            token(h, input[i].token);
            sequence(h, input[i].sequence);
            require(input[i].position >= 0 &&
                        static_cast<uint32_t>(input[i].position) < h->context_tokens,
                    "token position outside context");
            if (input[i].logits)
                ++rows;
        }
        require(rows <= std::numeric_limits<size_t>::max() / static_cast<size_t>(h->vocab_size),
                "logits buffer size overflow");
        require(output_count == rows * static_cast<size_t>(h->vocab_size) &&
                    (output || !output_count),
                "wrong logits output buffer size");
        if (!count)
            return 0;
        // 栈上 batch 只借用本次调用的 vector；没有跨调用的裸数据指针。
        std::vector<llama_token> tokens(count);
        std::vector<llama_pos> positions(count);
        std::vector<llama_seq_id> seqs(count);
        std::vector<llama_seq_id *> seq_ptrs(count);
        std::vector<int32_t> n_seqs(count, 1);
        std::vector<int8_t> logits(count);
        for (size_t i = 0; i < count; ++i) {
            tokens[i] = input[i].token;
            positions[i] = input[i].position;
            seqs[i] = input[i].sequence;
            seq_ptrs[i] = &seqs[i];
            logits[i] = input[i].logits != 0;
        }
        llama_batch batch{checked_i32(count), tokens.data(),   nullptr,      positions.data(),
                          n_seqs.data(),      seq_ptrs.data(), logits.data()};
        const int32_t status = llama_decode(h->context.get(), batch);
        if (status != 0)
            throw std::runtime_error("llama_decode failed with code " + std::to_string(status));
        size_t row = 0;
        for (size_t i = 0; i < count; ++i)
            if (input[i].logits) {
                const float *values = llama_get_logits_ith(h->context.get(), checked_i32(i));
                require(values != nullptr, "llama returned no requested logits row");
                std::copy_n(values, h->vocab_size,
                            output + row++ * static_cast<size_t>(h->vocab_size));
            }
        return 0;
    });
}
int32_t msgl_copy_seq(msgl_handle *h, int32_t src, int32_t dst, int32_t end) noexcept {
    return protect([&]() -> int32_t {
        valid(h);
        sequence(h, src);
        sequence(h, dst);
        require(end >= 0 && static_cast<uint32_t>(end) <= h->context_tokens,
                "invalid prefix length");
        require(src != dst, "KV copy requires distinct source and destination");
        auto memory = llama_get_memory(h->context.get());
        require(llama_memory_seq_pos_max(memory, dst) < 0, "KV copy destination must be empty");
        if (end == 0)
            return 0;
        require(llama_memory_seq_pos_min(memory, src) == 0 &&
                    llama_memory_seq_pos_max(memory, src) >= end - 1,
                "source sequence does not contain the requested prefix");
        llama_memory_seq_cp(memory, src, dst, 0, end);
        require(llama_memory_seq_pos_min(memory, dst) == 0 &&
                    llama_memory_seq_pos_max(memory, dst) >= end - 1,
                "KV prefix copy failed");
        return 0;
    });
}
int32_t msgl_remove_seq(msgl_handle *h, int32_t seq) noexcept {
    return protect([&]() -> int32_t {
        valid(h);
        sequence(h, seq);
        require(llama_memory_seq_rm(llama_get_memory(h->context.get()), seq, -1, -1),
                "KV sequence removal failed");
        return 0;
    });
}
int32_t msgl_apply_chat_template(msgl_handle *h, const msgl_chat_message *messages, size_t count,
                                 int32_t add_generation_prompt, char *output, size_t capacity,
                                 size_t *required) noexcept {
    return protect([&]() -> int32_t {
        valid(h);
        require(required && (messages || !count) && (output || !capacity), "invalid chat buffer");
        const char *tmpl = llama_model_chat_template(h->model.get(), nullptr);
        require(tmpl && tmpl[0], "GGUF has no chat template; use a text completion prompt");
        std::vector<llama_chat_message> chat;
        chat.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            require(messages[i].role && messages[i].content, "null chat role or content");
            chat.push_back({messages[i].role, messages[i].content});
        }
        const int32_t n =
            llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_generation_prompt != 0,
                                      output, checked_i32(capacity));
        require(n >= 0, "GGUF chat template is not supported by llama_chat_apply_template");
        *required = static_cast<size_t>(n);
        return *required > capacity ? 1 : 0;
    });
}
}
