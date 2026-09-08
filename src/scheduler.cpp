#include "minisgl/scheduler.hpp"
#include "minisgl/cache.hpp"
#include "minisgl/sampling.hpp"

#include <algorithm>
#include <deque>
#include <limits>
#include <list>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace minisgl {
namespace {
enum class SequenceOwner { Free, Active, Cache, Transient, Quarantined };
struct Pending {
    GenerationRequest request;
    std::vector<Token> prompt;
    std::size_t reservation = 0;
    bool cancelled = false;
};
struct Active {
    Pending input;
    SequenceId sequence;
    std::size_t prefilled = 0;
    std::size_t generated = 0;
    std::size_t cached = 0;
    std::size_t computed_prefill = 0;
    Token last_token = -1;
    std::mt19937_64 rng;
    Active(Pending pending, SequenceId id)
        : input(std::move(pending)), sequence(id), rng(input.request.sampling.seed) {}
};

Event finish_event(const Pending &pending, const std::string &reason,
                   const std::string &error = {}) {
    Event event;
    event.request_id = pending.request.id;
    event.kind = EventKind::Finished;
    event.finish_reason = reason;
    event.error = error;
    event.prompt_tokens = pending.prompt.size();
    return event;
}
Event active_event(const Active &request, const std::string &reason,
                   const std::string &error = {}) {
    auto event = finish_event(request.input, reason, error);
    event.completion_tokens = request.generated;
    event.cached_tokens = request.cached;
    event.prefill_tokens = request.computed_prefill;
    return event;
}
} // namespace

struct Scheduler::Impl {
    Engine &engine;
    SchedulerConfig config;
    std::deque<Pending> pending;
    // list 保持活跃请求地址与次序稳定，完成一个请求不会重排其他请求。
    std::list<std::unique_ptr<Active>> active;
    std::unordered_set<RequestId> live_ids;
    PrefixCache cache;
    std::unordered_map<SequenceId, std::size_t> cache_lengths;
    std::vector<SequenceOwner> owners;
    std::vector<SequenceId> free_ids;
    SchedulerStats counters;
    std::size_t reserved = 0;
    std::size_t cached_tokens = 0;
    std::size_t prefill_cursor = 0;
    std::uint64_t next_id = 1;
    bool healthy = true;

    Impl(Engine &backend, SchedulerConfig options) : engine(backend), config(options) {
        validate_config(config);
        owners.assign(config.max_sequences, SequenceOwner::Free);
        free_ids.reserve(config.max_sequences);
        for (std::size_t i = config.max_sequences; i > 0; --i)
            free_ids.push_back(static_cast<SequenceId>(i - 1));
    }
    ~Impl() {
        for (std::size_t i = 0; i < owners.size(); ++i) {
            if (owners[i] == SequenceOwner::Free)
                continue;
            try {
                engine.remove_sequence(static_cast<SequenceId>(i));
            } catch (...) { /* 析构不能掩盖调用者异常。 */
            }
        }
    }

    SequenceId acquire(SequenceOwner owner) {
        if (free_ids.empty())
            throw std::logic_error("sequence pool exhausted");
        const auto id = free_ids.back();
        free_ids.pop_back();
        owners.at(static_cast<std::size_t>(id)) = owner;
        return id;
    }
    void release(SequenceId id) {
        auto &owner = owners.at(static_cast<std::size_t>(id));
        if (owner == SequenceOwner::Free)
            throw std::logic_error("double release of sequence ID");
        // 必须先释放物理引用再复用 ID；失败时由 fail_all 关闭整个调度器。
        engine.remove_sequence(id);
        owner = SequenceOwner::Free;
        free_ids.push_back(id);
    }
    bool evict_one() {
        auto id = cache.evict_lru();
        if (!id)
            return false;
        auto entry = cache_lengths.find(*id);
        if (entry == cache_lengths.end())
            throw std::logic_error("cache ownership index is inconsistent");
        cached_tokens -= entry->second;
        cache_lengths.erase(entry);
        release(*id);
        return true;
    }
    bool room_for(std::size_t tokens) {
        while (tokens > config.max_total_tokens - reserved - cached_tokens || free_ids.empty()) {
            if (!evict_one())
                return false;
        }
        return true;
    }

    void maybe_cache(Active &request) {
        const auto &prompt = request.input.prompt;
        if (request.prefilled != prompt.size() || !config.max_cache_entries ||
            !config.max_cache_tokens || prompt.size() > config.max_cache_tokens ||
            prompt.size() > config.max_total_tokens - reserved)
            return;
        // 现有更长 prompt 同样能供应全部 KV，无需重复保存同一前缀。
        auto existing = cache.find(prompt, prompt.size());
        if (existing && existing->matched_tokens == prompt.size())
            return;
        while (cache.size() >= config.max_cache_entries || free_ids.empty() ||
               prompt.size() > config.max_cache_tokens - cached_tokens ||
               prompt.size() > config.max_total_tokens - reserved - cached_tokens) {
            if (!evict_one())
                return;
        }
        auto id = acquire(SequenceOwner::Transient);
        engine.copy_sequence(request.sequence, id, prompt.size());
        const auto replaced = cache.insert(prompt, id);
        if (replaced)
            throw std::logic_error("unexpected duplicate cache insertion");
        cache_lengths.emplace(id, prompt.size());
        cached_tokens += prompt.size();
        owners[static_cast<std::size_t>(id)] = SequenceOwner::Cache;
    }

    void count_finish(const std::string &reason) {
        ++counters.total_finished;
        if (reason == "cancelled")
            ++counters.total_cancelled;
        if (reason == "error")
            ++counters.total_failed;
    }
    void finish(Active *request, const std::string &reason, std::vector<Event> &events) {
        auto it = std::find_if(active.begin(), active.end(), [request](const auto &candidate) {
            return candidate.get() == request;
        });
        if (it == active.end())
            throw std::logic_error("unknown active request");
        reserved -= request->input.reservation;
        // 完成后归还生成预留空间，可尝试留下仅含 prompt 的独立 KV 引用。
        if (reason == "stop" || reason == "length")
            maybe_cache(*request);
        release(request->sequence);
        events.push_back(active_event(*request, reason));
        live_ids.erase(request->input.request.id);
        active.erase(it);
        count_finish(reason);
    }

    void fail_all(const std::string &message, std::vector<Event> &events) {
        // 前向可能部分写入 KV。无法回滚时 fail-closed，绝不在未知状态续跑。
        healthy = false;
        // 先清理后端，再分配错误事件；内存不足也不能阻止 KV 清理。
        free_ids.clear();
        for (std::size_t i = 0; i < owners.size(); ++i) {
            if (owners[i] != SequenceOwner::Free) {
                try {
                    engine.remove_sequence(static_cast<SequenceId>(i));
                    owners[i] = SequenceOwner::Free;
                } catch (...) {
                    // 不能把清理失败伪装成释放成功；析构时再尝试一次。
                    owners[i] = SequenceOwner::Quarantined;
                }
            }
            if (owners[i] == SequenceOwner::Free)
                free_ids.push_back(static_cast<SequenceId>(i));
        }
        cache.clear();
        cache_lengths.clear();
        cached_tokens = 0;
        reserved = 0;
        struct ClearRequests {
            Impl &state;
            ~ClearRequests() {
                state.active.clear();
                state.pending.clear();
                state.live_ids.clear();
            }
        } clear_requests{*this};
        for (const auto &request : active) {
            count_finish("error");
            events.push_back(active_event(*request, "error", message));
        }
        for (const auto &request : pending) {
            count_finish("error");
            events.push_back(finish_event(request, "error", message));
        }
    }

    void reap(std::vector<Event> &events) {
        for (auto it = pending.begin(); it != pending.end();) {
            if (!it->cancelled && it->request.max_new_tokens != 0) {
                ++it;
                continue;
            }
            const std::string reason = it->cancelled ? "cancelled" : "length";
            events.push_back(finish_event(*it, reason));
            live_ids.erase(it->request.id);
            count_finish(reason);
            it = pending.erase(it);
        }
        for (auto it = active.begin(); it != active.end();) {
            auto *request = (it++)->get();
            if (request->input.cancelled)
                finish(request, "cancelled", events);
        }
    }
    void admit() {
        while (!pending.empty() && active.size() < config.max_running_requests) {
            if (!room_for(pending.front().reservation))
                break;
            // 驱逐完成后再匹配，借用的 source 不会在复制前被驱逐。
            auto match = cache.find(pending.front().prompt, pending.front().prompt.size() - 1);
            const auto id = acquire(SequenceOwner::Active);
            // 先分配 list 节点和 Active 对象，成功后才移走队头。若分配失败，
            // fail_all 仍能在 pending 中找到已接受请求并交付唯一完成事件。
            active.emplace_back();
            try {
                active.back() = std::make_unique<Active>(std::move(pending.front()), id);
            } catch (...) {
                active.pop_back();
                throw;
            }
            pending.pop_front();
            reserved += active.back()->input.reservation;
            auto &admitted = *active.back();
            if (match) {
                engine.copy_sequence(match->sequence, id, match->matched_tokens);
                admitted.cached = match->matched_tokens;
                admitted.prefilled = match->matched_tokens;
                counters.cache_hit_tokens += match->matched_tokens;
            }
        }
    }

    void advance(std::vector<Event> &events) {
        reap(events);
        admit();
        struct Job {
            Active *request;
            std::size_t count;
            bool prefill;
            bool logits;
        };
        std::vector<BatchToken> batch;
        std::vector<Job> jobs;
        std::vector<Active *> prefills;
        batch.reserve(config.max_batch_tokens);
        for (auto &pointer : active) {
            auto &request = *pointer;
            if (request.prefilled < request.input.prompt.size()) {
                prefills.push_back(&request);
            } else {
                if (!request.generated)
                    throw std::logic_error("prefill completed without sampled token");
                const auto position = request.input.prompt.size() + request.generated - 1;
                batch.push_back({request.last_token, position, request.sequence, true});
                jobs.push_back({&request, 1, false, true});
            }
        }
        // active <= batch budget，因此每个 prefill 都能至少前进一步。
        // 余量按轮转顺序发放，长 prompt 不能占掉 decode，也不会饿死短 prompt。
        if (!prefills.empty()) {
            std::vector<std::size_t> quotas(prefills.size(), 1);
            auto remaining = config.max_batch_tokens - batch.size() - prefills.size();
            const auto start = prefill_cursor % prefills.size();
            for (std::size_t offset = 0; offset < prefills.size(); ++offset) {
                const auto index = (start + offset) % prefills.size();
                const auto &request = *prefills[index];
                const auto unfilled = request.input.prompt.size() - request.prefilled;
                const auto limit = std::min(unfilled, config.prefill_chunk_size);
                const auto extra = std::min(remaining, limit - 1);
                quotas[index] += extra;
                remaining -= extra;
            }
            prefill_cursor = (start + 1) % prefills.size();
            for (std::size_t i = 0; i < prefills.size(); ++i) {
                auto &request = *prefills[i];
                const auto count = quotas[i];
                const bool complete = request.prefilled + count == request.input.prompt.size();
                for (std::size_t offset = 0; offset < count; ++offset) {
                    const auto position = request.prefilled + offset;
                    batch.push_back({request.input.prompt[position], position, request.sequence,
                                     complete && offset + 1 == count});
                }
                jobs.push_back({&request, count, true, complete});
            }
        }
        if (batch.empty())
            return;
        ++counters.forward_calls;
        auto logits = engine.forward(batch);
        const auto expected = static_cast<std::size_t>(
            std::count_if(jobs.begin(), jobs.end(), [](const Job &job) { return job.logits; }));
        if (logits.size() != expected)
            throw std::runtime_error("backend returned an incorrect logits row count");
        // 完整前向成功后才推进逻辑位置，避免半提交的调度状态。
        for (const auto &job : jobs) {
            if (job.prefill) {
                job.request->prefilled += job.count;
                job.request->computed_prefill += job.count;
                counters.prefill_tokens += job.count;
            } else {
                counters.decode_tokens += job.count;
            }
        }
        std::size_t row = 0;
        for (const auto &job : jobs) {
            if (!job.logits)
                continue;
            auto &request = *job.request;
            if (job.prefill)
                maybe_cache(request);
            const auto token =
                sample_token(logits[row++], request.input.request.sampling, request.rng);
            request.last_token = token;
            ++request.generated;
            const bool eos = !request.input.request.ignore_eos && engine.is_eog(token);
            if (!eos) {
                auto event = active_event(request, {});
                event.kind = EventKind::Token;
                event.token = token;
                event.text = engine.piece(token);
                events.push_back(std::move(event));
            }
            if (eos)
                finish(&request, "stop", events);
            else if (request.generated >= request.input.request.max_new_tokens)
                finish(&request, "length", events);
        }
    }
};

Scheduler::Scheduler(Engine &engine, SchedulerConfig config)
    : impl_(std::make_unique<Impl>(engine, config)) {}
Scheduler::~Scheduler() = default;

RequestId Scheduler::submit(GenerationRequest request) {
    auto &state = *impl_;
    if (!state.healthy)
        throw std::runtime_error("scheduler is unavailable after backend failure");
    if (state.pending.size() >= state.config.max_pending_requests)
        throw std::runtime_error("pending request queue is full");
    validate_sampling(request.sampling);
    if (request.prompt.empty())
        throw std::invalid_argument("prompt must not be empty");
    if (!request.id.empty() && state.live_ids.count(request.id))
        throw std::invalid_argument("request ID is already active or queued");
    auto prompt = state.engine.tokenize(request.prompt);
    if (prompt.empty())
        throw std::invalid_argument("tokenized prompt must not be empty");
    if (std::any_of(prompt.begin(), prompt.end(), [](Token token) { return token < 0; }))
        throw std::invalid_argument("tokenizer returned a negative token ID");
    if (prompt.size() > state.config.max_sequence_tokens)
        throw std::invalid_argument("prompt exceeds the per-sequence context limit");
    std::size_t reservation = 0;
    if (request.max_new_tokens) {
        const auto generated_kv = request.max_new_tokens - 1;
        if (generated_kv > state.config.max_sequence_tokens - prompt.size() ||
            prompt.size() > state.config.max_total_tokens ||
            generated_kv > state.config.max_total_tokens - prompt.size())
            throw std::invalid_argument("request exceeds the available KV token budget");
        reservation = prompt.size() + generated_kv;
    }
    if (request.id.empty()) {
        do {
            if (state.next_id == std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("automatic request IDs exhausted");
            request.id = "req-" + std::to_string(state.next_id++);
        } while (state.live_ids.count(request.id));
    }
    const auto id = request.id;
    state.live_ids.insert(id);
    try {
        state.pending.push_back({std::move(request), std::move(prompt), reservation, false});
    } catch (...) {
        state.live_ids.erase(id);
        throw;
    }
    ++state.counters.total_submitted;
    return id;
}

bool Scheduler::cancel(const RequestId &id) {
    auto &state = *impl_;
    for (auto &request : state.pending) {
        if (request.request.id == id) {
            request.cancelled = true;
            return true;
        }
    }
    for (auto &request : state.active) {
        if (request->input.request.id == id) {
            request->input.cancelled = true;
            return true;
        }
    }
    return false;
}

std::vector<Event> Scheduler::step() {
    std::vector<Event> events;
    if (!impl_->healthy)
        return events;
    try {
        impl_->advance(events);
    } catch (const std::exception &error) {
        impl_->fail_all(error.what(), events);
    } catch (...) {
        impl_->fail_all("unknown backend failure", events);
    }
    return events;
}

bool Scheduler::idle() const { return impl_->pending.empty() && impl_->active.empty(); }
SchedulerStats Scheduler::stats() const {
    auto result = impl_->counters;
    result.active_requests = impl_->active.size();
    result.queued_requests = impl_->pending.size();
    result.cache_entries = impl_->cache.size();
    result.cache_tokens = impl_->cached_tokens;
    result.reserved_tokens = impl_->reserved;
    result.free_sequences = impl_->free_ids.size();
    result.quarantined_sequences = static_cast<std::size_t>(
        std::count(impl_->owners.begin(), impl_->owners.end(), SequenceOwner::Quarantined));
    result.healthy = impl_->healthy;
    return result;
}

void Scheduler::assert_invariants() const {
    const auto &state = *impl_;
    const auto require = [](bool condition, const char *message) {
        if (!condition)
            throw std::logic_error(message);
    };
    state.cache.assert_invariants();
    require(state.pending.size() <= state.config.max_pending_requests, "pending capacity");
    require(state.active.size() <= state.config.max_running_requests, "active capacity");
    require(state.cache.size() <= state.config.max_cache_entries, "cache entry capacity");
    require(state.cached_tokens <= state.config.max_cache_tokens, "cache token capacity");
    require(state.reserved <= state.config.max_total_tokens &&
                state.cached_tokens <= state.config.max_total_tokens - state.reserved,
            "total KV reservation capacity");
    require(state.cache_lengths.size() == state.cache.size(), "cache sequence count");
    std::vector<bool> seen(state.owners.size(), false);
    std::unordered_set<RequestId> ids;
    std::size_t reserved = 0;
    std::size_t cached = 0;
    const auto mark = [&](SequenceId id, SequenceOwner owner) {
        require(id >= 0 && static_cast<std::size_t>(id) < seen.size(), "sequence ID bounds");
        require(!seen[static_cast<std::size_t>(id)], "duplicate sequence ownership");
        require(state.owners[static_cast<std::size_t>(id)] == owner, "sequence owner mismatch");
        seen[static_cast<std::size_t>(id)] = true;
    };
    for (auto id : state.free_ids)
        mark(id, SequenceOwner::Free);
    for (std::size_t i = 0; i < state.owners.size(); ++i) {
        if (state.owners[i] == SequenceOwner::Quarantined) {
            require(!state.healthy, "healthy scheduler contains quarantined sequence");
            mark(static_cast<SequenceId>(i), SequenceOwner::Quarantined);
        }
    }
    for (const auto &request : state.active) {
        mark(request->sequence, SequenceOwner::Active);
        reserved += request->input.reservation;
        require(ids.insert(request->input.request.id).second, "duplicate active request ID");
        require(request->prefilled <= request->input.prompt.size(), "prefill position");
        require(request->generated < request->input.request.max_new_tokens, "unfinished length");
        require(request->prefilled == request->cached + request->computed_prefill,
                "prefill token accounting");
    }
    for (const auto &entry : state.cache_lengths) {
        mark(entry.first, SequenceOwner::Cache);
        cached += entry.second;
    }
    for (const auto &request : state.pending)
        require(ids.insert(request.request.id).second, "duplicate pending request ID");
    require(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }),
            "untracked sequence ownership");
    require(reserved == state.reserved && cached == state.cached_tokens, "token accounting");
    require(ids == state.live_ids, "request ID ownership");
}
} // namespace minisgl
