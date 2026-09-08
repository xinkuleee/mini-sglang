#include "minisgl/cache.hpp"
#include "minisgl/sampling.hpp"
#include "minisgl/scheduler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace minisgl;
namespace {
void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class Function> void throws(Function &&function) {
    bool caught = false;
    try {
        function();
    } catch (const std::exception &) {
        caught = true;
    }
    check(caught, "expected an exception");
}

// 测试后端只验证 runtime 协议：逐位置写入、KV 复制及模型输出隔离。
// 它仅链接到测试目标，CLI/服务始终要求真实 GGUF 模型后端。
class FakeEngine final : public Engine {
  public:
    std::unordered_map<SequenceId, std::vector<Token>> sequences;
    std::vector<std::vector<BatchToken>> batches;
    struct Copy {
        SequenceId src;
        SequenceId dst;
        std::size_t length;
    };
    std::vector<Copy> copies;
    bool eos = false;
    bool stochastic = false;
    bool fail_forward = false;
    bool fail_copy = false;
    bool wrong_rows = false;
    bool fail_remove_once = false;
    std::size_t fail_piece_call = 0;
    std::size_t piece_calls = 0;

    std::vector<Token> tokenize(const std::string &text) override {
        std::vector<Token> tokens;
        for (unsigned char character : text)
            tokens.push_back(character);
        return tokens;
    }
    std::string piece(Token token) override {
        if (++piece_calls == fail_piece_call)
            throw std::runtime_error("injected piece failure");
        return std::string(1, static_cast<char>(token));
    }
    bool is_eog(Token token) const override { return token == 0; }
    std::vector<std::vector<float>> forward(std::span<const BatchToken> batch) override {
        check(!batch.empty(), "empty backend batch");
        batches.emplace_back(batch.begin(), batch.end());
        std::vector<std::vector<float>> output;
        for (const auto &item : batch) {
            auto &sequence = sequences[item.sequence];
            check(item.position == sequence.size(), "noncontiguous or contaminated sequence KV");
            sequence.push_back(item.token);
            if (fail_forward)
                throw std::runtime_error("injected partial backend failure");
            if (!item.logits)
                continue;
            std::vector<float> logits(256, -1000.0f);
            std::uint64_t hash = 2166136261u;
            for (auto token : sequence)
                hash = (hash ^ static_cast<unsigned>(token)) * 16777619u;
            const auto selected = eos ? 0 : 65 + static_cast<int>(hash % 26);
            logits[static_cast<std::size_t>(selected)] = 8.0f;
            if (stochastic && !eos) {
                logits[65 + (selected - 64) % 26] = 7.5f;
                logits[65 + (selected - 63) % 26] = 7.0f;
            }
            output.push_back(std::move(logits));
        }
        if (wrong_rows)
            output.push_back({0.0f});
        return output;
    }
    void copy_sequence(SequenceId src, SequenceId dst, std::size_t end) override {
        const auto &source = sequences.at(src);
        check(end <= source.size(), "copy exceeds source KV");
        check(sequences.find(dst) == sequences.end(), "copy destination is not empty");
        sequences[dst] = std::vector<Token>(source.begin(), source.begin() + end);
        copies.push_back({src, dst, end});
        if (fail_copy)
            throw std::runtime_error("injected copy failure");
    }
    void remove_sequence(SequenceId sequence) override {
        if (fail_remove_once) {
            fail_remove_once = false;
            throw std::runtime_error("injected remove failure");
        }
        sequences.erase(sequence);
    }
};

SchedulerConfig config_without_cache() {
    SchedulerConfig config;
    config.max_running_requests = 3;
    config.max_pending_requests = 8;
    config.max_batch_tokens = 4;
    config.prefill_chunk_size = 3;
    config.max_sequence_tokens = 100;
    config.max_total_tokens = 256;
    config.max_sequences = 6;
    config.max_cache_entries = 0;
    config.max_cache_tokens = 0;
    return config;
}
GenerationRequest request(std::string id, std::string prompt, std::size_t length = 4) {
    GenerationRequest result;
    result.id = std::move(id);
    result.prompt = std::move(prompt);
    result.max_new_tokens = length;
    return result;
}
std::vector<Event> drain(Scheduler &scheduler) {
    std::vector<Event> events;
    std::size_t ticks = 0;
    while (!scheduler.idle()) {
        check(++ticks < 10000, "scheduler did not make progress");
        auto step = scheduler.step();
        events.insert(events.end(), step.begin(), step.end());
        scheduler.assert_invariants();
    }
    return events;
}
const Event &finished(const std::vector<Event> &events, const std::string &id) {
    const Event *result = nullptr;
    for (const auto &event : events) {
        if (event.kind == EventKind::Finished && event.request_id == id) {
            check(result == nullptr, "request completed twice");
            result = &event;
        }
    }
    check(result != nullptr, "request never completed");
    return *result;
}
std::string output(const std::vector<Event> &events, const std::string &id) {
    std::string text;
    for (const auto &event : events)
        if (event.kind == EventKind::Token && event.request_id == id)
            text += event.text;
    return text;
}

void borrowed_span_inputs() {
    // Subviews exclude adjacent storage; inserted cache tokens and recorded batches own copies.
    SamplingParams options;
    std::mt19937_64 rng(9);
    const std::array scores{100.0f, 0.0f, 3.0f, 1.0f, 100.0f};
    check(sample_token(std::span{scores}.subspan(1, 3), options, rng) == 1,
          "sampling read outside the logits view");

    PrefixCache cache;
    {
        std::array<Token, 5> prompt{99, 1, 2, 3, 88};
        cache.insert(std::span{prompt}.subspan(1, 3), 7);
        prompt.fill(0);
    }
    const std::array<Token, 3> query{1, 2, 4};
    const auto match = cache.find(query, query.size());
    check(match && match->sequence == 7 && match->matched_tokens == 2,
          "cache retained a borrowed view instead of copying its tokens");
    cache.assert_invariants();

    FakeEngine engine;
    {
        std::array<BatchToken, 4> batch{
            {{-1, 99, -1, true}, {65, 0, 0, false}, {66, 1, 0, true}, {-1, 99, -1, true}}};
        const auto logits = engine.forward(std::span{batch}.subspan(1, 2));
        check(logits.size() == 1, "batch view changed requested logits rows");
        batch[1].token = 0;
    }
    check(engine.sequences.at(0) == std::vector<Token>({65, 66}) &&
              engine.batches.front().size() == 2 && engine.batches.front()[0].token == 65,
          "backend retained a borrowed batch or read outside its view");
}

void sampling_semantics() {
    SamplingParams options;
    std::mt19937_64 rng(123);
    auto same_rng = rng;
    check(sample_token(std::array{2.0f, 2.0f, -1.0f}, options, rng) == 0, "greedy tie");
    check(rng() == same_rng(), "greedy consumed randomness");
    const auto inf = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    throws([&] { sample_token({}, options, rng); });
    throws([&] { sample_token(std::array{-inf, -inf}, options, rng); });
    throws([&] { sample_token(std::array{nan, 1.0f}, options, rng); });
    throws([&] { sample_token(std::array{inf, 1.0f}, options, rng); });
    check(sample_token(std::array{-inf, -3.0f}, options, rng) == 1, "negative infinity masking");
    options.temperature = -1;
    throws([&] { validate_sampling(options); });
    options.temperature = inf;
    throws([&] { validate_sampling(options); });
    options.temperature = 1;
    options.top_p = 0;
    throws([&] { validate_sampling(options); });
    options.top_p = 1.1f;
    throws([&] { validate_sampling(options); });
    options.top_p = 1;
    options.top_k = 1;
    for (int i = 0; i < 100; ++i)
        check(sample_token(std::array{0.0f, 3.0f, 2.0f}, options, rng) == 1, "top-k filter");
    options.top_k = 0;
    options.top_p = 0.1f;
    for (int i = 0; i < 100; ++i)
        check(sample_token(std::array{0.0f, 3.0f, 2.0f}, options, rng) == 1,
              "top-p minimal prefix");
    options.top_p = 1;
    options.temperature = std::numeric_limits<float>::min();
    check(sample_token(std::array{0.0f, 3.0f, 2.0f}, options, rng) == 1,
          "small temperature stability");
    options.temperature = 1;
    std::mt19937_64 first(7), second(7), unrelated(8);
    for (int i = 0; i < 1000; ++i) {
        const auto token = sample_token(std::array{0.0f, 1.0f, 2.0f}, options, first);
        sample_token(std::array{2.0f, 1.0f, 0.0f}, options, unrelated);
        check(token == sample_token(std::array{0.0f, 1.0f, 2.0f}, options, second),
              "request RNG isolation");
    }
}

void radix_split_eviction() {
    PrefixCache cache;
    cache.insert(std::array{1, 2, 3}, 10);
    cache.insert(std::array{1, 2, 4}, 11);
    check(cache.size() == 2 && cache.token_count() == 4 && cache.node_count() == 4,
          "radix split does not share compressed prefix");
    auto match = cache.find(std::array{1, 2, 3, 5}, 4);
    check(match && match->matched_tokens == 3 && match->sequence == 10, "longest prefix");
    match = cache.find(std::array{1, 2, 3}, 2);
    check(match && match->matched_tokens == 2, "match length limit");
    cache.insert(std::array{1}, 12);
    cache.assert_invariants();
    check(cache.token_count() == 4, "shorter terminal should not duplicate tokens");
    throws([&] { cache.insert(std::array{8}, 12); });
    throws([&] { cache.insert({}, 13); });
    throws([&] { cache.insert(std::array{8}, -1); });
    check(!cache.find(std::array{9}, 5), "unrelated prefixes matched");
    auto old = cache.insert(std::array{1, 2, 4}, 13);
    check(old && *old == 11, "replacement did not return old owner");
    check(cache.erase_sequence(12), "erase shorter terminal");
    check(!cache.erase_sequence(12), "double erase");
    cache.assert_invariants();
    cache.find(std::array{1, 2, 3}, 3);
    check(cache.evict_lru() == std::optional<SequenceId>(13), "LRU refresh");
    check(cache.node_count() == 2 && cache.token_count() == 3, "unary path not recompressed");
    cache.assert_invariants();
    check(cache.evict_lru() == std::optional<SequenceId>(10), "final eviction");
    check(cache.node_count() == 1 && cache.token_count() == 0 && !cache.evict_lru(),
          "empty cache invariants");
    cache.assert_invariants();
}

void radix_randomized_oracle() {
    // 朴素完整 prompt 表作为独立 oracle，不复刻 radix 的分裂/压缩算法。
    // 固定种子覆盖交错命中、替换、LRU 和删除；失败可重复且无需外部框架。
    struct Entry {
        std::vector<Token> tokens;
        std::size_t touched;
    };
    PrefixCache cache;
    std::map<SequenceId, Entry> entries;
    std::mt19937_64 random(0x6d696e6973676cULL);
    std::size_t clock = 0;
    SequenceId next_sequence = 0;
    const auto random_tokens = [&] {
        std::vector<Token> tokens(1 + random() % 10);
        for (auto &token : tokens)
            token = static_cast<Token>(random() % 6);
        return tokens;
    };
    const auto random_entry = [&] {
        auto entry = entries.begin();
        std::advance(entry, static_cast<std::ptrdiff_t>(random() % entries.size()));
        return entry;
    };
    const auto common_prefix = [](const std::vector<Token> &left, const std::vector<Token> &right,
                                  std::size_t limit) {
        const auto end = std::min({left.size(), right.size(), limit});
        std::size_t count = 0;
        while (count < end && left[count] == right[count])
            ++count;
        return count;
    };
    for (std::size_t step = 0; step < 3000; ++step) {
        const auto operation = random() % 10;
        if (step % 257 == 256) {
            cache.clear();
            cache.clear();
            entries.clear();
        } else if (entries.size() >= 32 || operation == 7) {
            std::optional<SequenceId> expected;
            if (!entries.empty()) {
                const auto oldest = std::min_element(
                    entries.begin(), entries.end(), [](const auto &left, const auto &right) {
                        return left.second.touched < right.second.touched;
                    });
                expected = oldest->first;
                entries.erase(oldest);
            }
            check(cache.evict_lru() == expected, "random oracle: LRU victim");
        } else if (operation < 4) {
            auto tokens = random_tokens();
            if (!entries.empty() && random() % 4 == 0)
                tokens = random_entry()->second.tokens;
            std::optional<SequenceId> expected;
            for (const auto &entry : entries) {
                if (entry.second.tokens == tokens) {
                    expected = entry.first;
                    break;
                }
            }
            const auto sequence = next_sequence++;
            check(cache.insert(tokens, sequence) == expected, "random oracle: replacement owner");
            if (expected)
                entries.erase(*expected);
            entries.emplace(sequence, Entry{std::move(tokens), ++clock});
        } else if (operation < 7) {
            auto tokens = random_tokens();
            if (!entries.empty() && random() % 2 == 0) {
                tokens = random_entry()->second.tokens;
                if (random() % 2 == 0)
                    tokens.push_back(static_cast<Token>(random() % 6));
            }
            const auto limit = static_cast<std::size_t>(random() % (tokens.size() + 2));
            std::size_t expected = 0;
            for (const auto &entry : entries)
                expected = std::max(expected, common_prefix(tokens, entry.second.tokens, limit));
            const auto match = cache.find(tokens, limit);
            check(static_cast<bool>(match) == (expected != 0), "random oracle: hit presence");
            if (match) {
                const auto source = entries.find(match->sequence);
                check(source != entries.end() && match->matched_tokens == expected &&
                          common_prefix(tokens, source->second.tokens, limit) == expected,
                      "random oracle: longest prefix and valid KV source");
                source->second.touched = ++clock;
            }
        } else if (operation == 8) {
            const auto sequence =
                entries.empty() || random() % 3 == 0 ? next_sequence + 100 : random_entry()->first;
            const bool existed = entries.erase(sequence) != 0;
            check(cache.erase_sequence(sequence) == existed, "random oracle: erase presence");
        } else if (!entries.empty()) {
            auto existing = random_entry();
            check(!cache.insert(existing->second.tokens, existing->first),
                  "random oracle: refreshing same sequence replaced ownership");
            existing->second.touched = ++clock;
        }
        cache.assert_invariants();
        check(cache.size() == entries.size(), "random oracle: terminal count");
        // 每个不同的非空前缀对应未压缩 trie 的一条边；压缩不改变总 token 数。
        // 这个较重的独立检查每十步执行，其余协议和不变量仍逐步校验。
        if (step % 10 == 0 || step == 2999) {
            std::set<std::vector<Token>> prefixes;
            for (const auto &entry : entries) {
                std::vector<Token> prefix;
                for (auto token : entry.second.tokens) {
                    prefix.push_back(token);
                    prefixes.insert(prefix);
                }
            }
            check(cache.token_count() == prefixes.size(),
                  "random oracle: shared prefix accounting");
        }
        check(cache.node_count() <= std::max<std::size_t>(1, 2 * entries.size()),
              "random oracle: uncompressed unary branches");
    }
}

void continuous_batch_budget() {
    FakeEngine engine;
    auto config = config_without_cache();
    Scheduler scheduler(engine, config);
    scheduler.submit(request("short", "a", 5));
    scheduler.submit(request("long", "abcdefghijkl", 3));
    scheduler.submit(request("medium", "1234567", 2));
    auto first = scheduler.step();
    check(output(first, "short").size() == 1, "short prompt did not start");
    scheduler.assert_invariants();
    check(engine.batches[0].size() == 4, "first batch quota");
    auto second = scheduler.step();
    check(engine.batches[1].front().sequence == engine.batches[0].front().sequence &&
              engine.batches[1].front().position == 1,
          "decode was not scheduled first");
    check(output(second, "short").size() == 1, "long prompt starved decode");
    auto rest = drain(scheduler);
    check(finished(rest, "long").completion_tokens == 3, "long request length");
    check(finished(rest, "medium").prefill_tokens == 7, "chunked prefill accounting");
    for (const auto &batch : engine.batches) {
        check(batch.size() <= config.max_batch_tokens, "forward exceeded token budget");
        std::map<SequenceId, std::size_t> per_sequence;
        for (const auto &token : batch)
            ++per_sequence[token.sequence];
        for (const auto &count : per_sequence)
            check(count.second <= config.prefill_chunk_size, "prefill chunk exceeded limit");
    }
    check(scheduler.stats().prefill_tokens == 20, "prompt computed multiple times");
    check(engine.sequences.empty(), "completed sequences leaked");
}

void cache_hit_and_generated_suffix() {
    FakeEngine engine;
    auto config = config_without_cache();
    config.max_cache_entries = 2;
    config.max_cache_tokens = 30;
    Scheduler scheduler(engine, config);
    scheduler.submit(request("first", "abcdef", 4));
    auto first = drain(scheduler);
    check(scheduler.stats().cache_entries == 1, "prompt not cached");
    check(scheduler.stats().cache_tokens == 6, "generated suffix entered prompt cache");
    check(engine.sequences.size() == 1 && engine.sequences.begin()->second.size() == 6,
          "cached backend sequence is not exactly prompt");
    scheduler.submit(request("repeat", "abcdef", 4));
    auto repeat = drain(scheduler);
    check(finished(repeat, "repeat").cached_tokens == 5, "last prompt token must be recomputed");
    check(finished(repeat, "repeat").prefill_tokens == 1, "cache hit recomputation");
    check(output(first, "first") == output(repeat, "repeat"), "cache changes generated output");
    scheduler.submit(request("branch", "abcXYZ", 1));
    auto branch = drain(scheduler);
    check(finished(branch, "branch").cached_tokens == 3, "partial radix branch reuse");
    check(scheduler.stats().cache_entries == 2, "branch cache not stored");
    scheduler.submit(request("third", "987654", 1));
    drain(scheduler);
    check(scheduler.stats().cache_entries == 2 && engine.sequences.size() == 2,
          "LRU cache capacity or KV cleanup");
}

void zero_eos_cancel_validation() {
    FakeEngine engine;
    auto config = config_without_cache();
    config.max_pending_requests = 2;
    Scheduler scheduler(engine, config);
    throws([&] { scheduler.submit(request("empty", "")); });
    throws([&] { scheduler.submit(request("large", "abc", 200)); });
    auto bad = request("bad", "a");
    bad.sampling.top_p = 0;
    throws([&] { scheduler.submit(bad); });
    scheduler.submit(request("zero", "abc", 0));
    throws([&] { scheduler.submit(request("zero", "abc")); });
    scheduler.submit(request("cancel-pending", "abc"));
    throws([&] { scheduler.submit(request("full", "abc")); });
    check(scheduler.cancel("cancel-pending"), "pending cancellation missing");
    check(!scheduler.cancel("unknown"), "unknown cancellation matched");
    auto events = drain(scheduler);
    check(finished(events, "zero").completion_tokens == 0 &&
              finished(events, "zero").prefill_tokens == 0 && engine.batches.empty(),
          "zero-output request executed model");
    check(finished(events, "cancel-pending").finish_reason == "cancelled", "cancel finish reason");
    engine.eos = true;
    scheduler.submit(request("eos", "a", 5));
    auto eos = drain(scheduler);
    check(output(eos, "eos").empty() && finished(eos, "eos").completion_tokens == 1 &&
              finished(eos, "eos").finish_reason == "stop",
          "EOS semantics");
    engine.eos = false;
    scheduler.submit(request("cancel-active", "abcdefghijklmnop", 5));
    scheduler.step();
    const auto batch_count = engine.batches.size();
    check(scheduler.cancel("cancel-active"), "active cancellation missing");
    auto cancelled = drain(scheduler);
    check(finished(cancelled, "cancel-active").finish_reason == "cancelled" &&
              engine.batches.size() == batch_count && engine.sequences.empty(),
          "cancelled request kept computing or leaked KV");
}

void admission_and_id_reuse() {
    FakeEngine engine;
    auto config = config_without_cache();
    config.max_total_tokens = 8;
    config.max_sequences = 3;
    Scheduler scheduler(engine, config);
    scheduler.submit(request("one", "abcd", 3));
    scheduler.submit(request("two", "efgh", 3));
    scheduler.step();
    check(scheduler.stats().active_requests == 1 && scheduler.stats().queued_requests == 1 &&
              scheduler.stats().reserved_tokens == 6,
          "admission did not reserve full future KV");
    auto rest = drain(scheduler);
    check(finished(rest, "two").completion_tokens == 3, "queued request did not progress");
    check(scheduler.stats().free_sequences == 3, "sequence pool leak");
    for (int i = 0; i < 100; ++i) {
        scheduler.submit(request("one", "a", 1));
        drain(scheduler);
    }
    check(scheduler.stats().free_sequences == 3 && engine.sequences.empty(), "ID reuse leak");
    config.max_batch_tokens = 2;
    throws([&] { Scheduler invalid(engine, config); });
    // 最后采样 token 不写 KV，因此 context=4 可接收 prompt=3、new=2。
    config.max_running_requests = 1;
    config.max_sequence_tokens = 4;
    Scheduler boundary(engine, config);
    boundary.submit(request("boundary", "abc", 2));
    check(finished(drain(boundary), "boundary").completion_tokens == 2, "KV boundary off by one");
    throws([&] { boundary.submit(request("overflow", "abc", 3)); });
}

void batch_and_rng_isolation() {
    auto run = [](bool together) {
        FakeEngine engine;
        engine.stochastic = true;
        auto config = config_without_cache();
        Scheduler scheduler(engine, config);
        auto a = request("a", "alpha", 12);
        a.sampling.temperature = 0.9f;
        a.sampling.seed = 123;
        auto b = request("b", "very-long-beta", 9);
        b.sampling.temperature = 0.8f;
        b.sampling.seed = 456;
        scheduler.submit(a);
        if (together)
            scheduler.submit(b);
        auto first = drain(scheduler);
        if (!together) {
            scheduler.submit(b);
            auto second = drain(scheduler);
            first.insert(first.end(), second.begin(), second.end());
        }
        return std::make_pair(output(first, "a"), output(first, "b"));
    };
    check(run(true) == run(false), "batching changed per-request model state or RNG");
}

void backend_failure_cleanup() {
    for (int failure = 0; failure < 4; ++failure) {
        FakeEngine engine;
        auto config = config_without_cache();
        config.max_cache_entries = 2;
        config.max_cache_tokens = 20;
        Scheduler scheduler(engine, config);
        if (failure == 0)
            engine.fail_forward = true;
        if (failure == 1)
            engine.wrong_rows = true;
        if (failure == 2)
            engine.fail_copy = true;
        if (failure == 3)
            engine.fail_remove_once = true;
        scheduler.submit(request("a", "a", 1));
        scheduler.submit(request("b", "b", 5));
        scheduler.submit(request("c", "c", 5));
        scheduler.submit(request("queued", "d", 5));
        auto events = drain(scheduler);
        for (const auto &id : {"a", "b", "c", "queued"})
            check(finished(events, id).finish_reason == "error",
                  "failure did not resolve every request");
        check(!scheduler.stats().healthy && scheduler.stats().total_failed == 4 &&
                  scheduler.stats().free_sequences == config.max_sequences &&
                  engine.sequences.empty(),
              "backend failure leaked state or stayed usable");
        check(scheduler.step().empty(), "failed scheduler emitted duplicate events");
        throws([&] { scheduler.submit(request("new", "new")); });
        scheduler.assert_invariants();
    }
    FakeEngine engine;
    {
        Scheduler scheduler(engine, config_without_cache());
        engine.fail_forward = true;
        engine.fail_remove_once = true;
        scheduler.submit(request("quarantine", "a", 2));
        auto events = drain(scheduler);
        check(finished(events, "quarantine").finish_reason == "error" &&
                  scheduler.stats().quarantined_sequences == 1 && engine.sequences.size() == 1,
              "failed cleanup must quarantine sequence instead of claiming it is free");
        scheduler.assert_invariants();
    }
    check(engine.sequences.empty(), "destructor did not retry quarantined cleanup");
    FakeEngine partial;
    partial.fail_piece_call = 2;
    Scheduler partial_scheduler(partial, config_without_cache());
    partial_scheduler.submit(request("done", "a", 1));
    partial_scheduler.submit(request("error", "b", 2));
    partial_scheduler.submit(request("unprocessed", "c", 2));
    auto partial_events = drain(partial_scheduler);
    check(finished(partial_events, "done").finish_reason == "length" &&
              finished(partial_events, "error").finish_reason == "error" &&
              finished(partial_events, "unprocessed").finish_reason == "error",
          "later batch failure changed already completed request");
    check(partial.sequences.empty(), "partial batch failure leaked KV");

    FakeEngine hit;
    auto hit_config = config_without_cache();
    hit_config.max_cache_entries = 2;
    hit_config.max_cache_tokens = 20;
    Scheduler hit_scheduler(hit, hit_config);
    hit_scheduler.submit(request("warmup", "abcdef", 1));
    drain(hit_scheduler);
    hit.fail_copy = true;
    hit_scheduler.submit(request("cache-copy", "abcdef", 2));
    auto hit_events = drain(hit_scheduler);
    check(finished(hit_events, "cache-copy").finish_reason == "error" && hit.sequences.empty(),
          "cache hit copy failure leaked source or destination");
}

void cache_pressure_and_same_batch_finish() {
    for (bool sequence_pressure : {false, true}) {
        FakeEngine engine;
        auto config = config_without_cache();
        config.max_running_requests = 1;
        config.max_sequences = sequence_pressure ? 2 : 6;
        config.max_total_tokens = sequence_pressure ? 100 : 8;
        config.max_cache_entries = 4;
        config.max_cache_tokens = 30;
        Scheduler scheduler(engine, config);
        for (const auto &prompt : {"abc", "def", "ghi", "jkl"}) {
            scheduler.submit(request(prompt, prompt, 1));
            auto events = drain(scheduler);
            check(finished(events, prompt).finish_reason == "length",
                  "cache pressure stalled admission");
            check(scheduler.stats().healthy, "cache pressure caused backend failure");
            check(scheduler.stats().cache_tokens <= config.max_total_tokens,
                  "cache pressure token budget");
        }
        check(engine.sequences.size() == scheduler.stats().cache_entries,
              "pressure eviction leaked KV");
    }
    FakeEngine engine;
    auto config = config_without_cache();
    Scheduler scheduler(engine, config);
    for (const auto &prompt : {"a", "b", "c"})
        scheduler.submit(request(prompt, prompt, 1));
    const auto events = drain(scheduler);
    check(engine.batches.size() == 1 && scheduler.stats().total_finished == 3 &&
              engine.sequences.empty(),
          "simultaneous batch completion");
    for (const auto &id : {"a", "b", "c"})
        check(finished(events, id).finish_reason == "length", "simultaneous finish reason");
}
} // namespace

int main() {
    const std::vector<std::pair<const char *, std::function<void()>>> tests = {
        {"borrowed span inputs and owned results", borrowed_span_inputs},
        {"sampling semantics", sampling_semantics},
        {"compressed radix split and LRU", radix_split_eviction},
        {"compressed radix randomized oracle (3000 operations)", radix_randomized_oracle},
        {"continuous batch fairness and token budget", continuous_batch_budget},
        {"prefix cache reuse and prompt-only snapshots", cache_hit_and_generated_suffix},
        {"zero output, EOS, cancellation, validation", zero_eos_cancel_validation},
        {"admission budget and sequence reuse", admission_and_id_reuse},
        {"batch and request RNG isolation", batch_and_rng_isolation},
        {"backend failure cleanup", backend_failure_cleanup},
        {"cache pressure and simultaneous completion", cache_pressure_and_same_batch_finish},
    };
    try {
        for (const auto &test : tests) {
            test.second();
            std::cout << "PASS " << test.first << '\n';
        }
    } catch (const std::exception &error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
    return 0;
}
