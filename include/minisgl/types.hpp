#pragma once

// 公共值类型只描述请求、事件和预算：传输层不必知道调度器内部状态，
// 后端也不拥有请求生命周期，使学习时能沿明确的数据边界阅读。
#include <cstddef>
#include <cstdint>
#include <string>

namespace minisgl {
using Token = std::int32_t;
using SequenceId = std::int32_t;
using RequestId = std::string;

struct SamplingParams {
    float temperature = 0.0f;
    std::size_t top_k = 0;
    float top_p = 1.0f;
    std::uint64_t seed = 0;
};

struct GenerationRequest {
    RequestId id;
    std::string prompt;
    std::size_t max_new_tokens = 32;
    SamplingParams sampling;
    bool ignore_eos = false;
};

enum class EventKind { Token, Finished };
struct Event {
    RequestId request_id;
    EventKind kind = EventKind::Finished;
    minisgl::Token token = -1;
    // Token 的文本是原始字节；UTF-8 字符可以跨多个 token。
    std::string text;
    std::string finish_reason;
    std::string error;
    std::size_t prompt_tokens = 0;
    std::size_t completion_tokens = 0;
    std::size_t cached_tokens = 0;
    std::size_t prefill_tokens = 0;
};

struct SchedulerConfig {
    std::size_t max_running_requests = 8;
    std::size_t max_pending_requests = 128;
    std::size_t max_batch_tokens = 256;
    std::size_t prefill_chunk_size = 64;
    std::size_t max_sequence_tokens = 2048;
    std::size_t max_total_tokens = 8192;
    std::size_t max_sequences = 16;
    std::size_t max_cache_entries = 8;
    std::size_t max_cache_tokens = 4096;
};

struct SchedulerStats {
    std::size_t active_requests = 0;
    std::size_t queued_requests = 0;
    std::size_t cache_entries = 0;
    std::size_t cache_tokens = 0;
    std::size_t reserved_tokens = 0;
    std::size_t free_sequences = 0;
    std::size_t quarantined_sequences = 0;
    std::size_t total_submitted = 0;
    std::size_t total_finished = 0;
    std::size_t total_cancelled = 0;
    std::size_t total_failed = 0;
    std::size_t prefill_tokens = 0;
    std::size_t decode_tokens = 0;
    std::size_t cache_hit_tokens = 0;
    std::size_t forward_calls = 0;
    bool healthy = true;
};

void validate_config(const SchedulerConfig &config);
} // namespace minisgl
