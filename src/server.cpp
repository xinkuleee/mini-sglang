// HTTP 线程只做解析和传递消息。唯一 owner 线程持有 Scheduler 并调用模型，
// 避免给 KV/FFI 加锁；有界入口和输出队列把慢客户端的内存影响限制在常数。
#include "minisgl/server.hpp"
#include "minisgl/llama_engine.hpp"
#include "minisgl/scheduler.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace minisgl {
// 不让 JSON 库逐 token 修复 UTF-8：那会把跨 token 的合法汉字变成替换字符。
std::string consume_utf8(std::string& pending, bool finish) {
    std::string output;
    std::size_t i = 0;
    while (i < pending.size()) {
        const auto first = static_cast<unsigned char>(pending[i]);
        std::size_t length = first < 0x80 ? 1 :
            (first >= 0xc2 && first <= 0xdf ? 2 :
             (first >= 0xe0 && first <= 0xef ? 3 :
              (first >= 0xf0 && first <= 0xf4 ? 4 : 0)));
        if (length == 0) { output += "\xef\xbf\xbd"; ++i; continue; }
        std::size_t valid_prefix = 1;
        while (valid_prefix < length && i + valid_prefix < pending.size()) {
            const auto byte = static_cast<unsigned char>(pending[i + valid_prefix]);
            if ((byte & 0xc0) != 0x80) break;
            if (valid_prefix == 1 && ((first == 0xe0 && byte < 0xa0) ||
                (first == 0xed && byte >= 0xa0) || (first == 0xf0 && byte < 0x90) ||
                (first == 0xf4 && byte >= 0x90))) break;
            ++valid_prefix;
        }
        if (valid_prefix < length) {
            if (!finish && i + valid_prefix == pending.size()) break;
            output += "\xef\xbf\xbd"; i += valid_prefix; continue;
        }
        output.append(pending, i, length); i += length;
    }
    pending.erase(0, i);
    return output;
}

namespace {
using nlohmann::json;
using namespace std::chrono_literals;
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }
class SignalHandlers {
public:
    SignalHandlers() {
        interrupted = 0;
        old_int_ = std::signal(SIGINT, on_signal);
        old_term_ = std::signal(SIGTERM, on_signal);
    }
    ~SignalHandlers() {
        if (old_int_ != SIG_ERR) std::signal(SIGINT, old_int_);
        if (old_term_ != SIG_ERR) std::signal(SIGTERM, old_term_);
    }
    SignalHandlers(const SignalHandlers&) = delete;
    SignalHandlers& operator=(const SignalHandlers&) = delete;
private:
    using Handler = void (*)(int);
    Handler old_int_, old_term_;
};
constexpr std::size_t max_pending = 64, max_events = 256;
json error_json(const std::string& message, const std::string& type) {
    return {{"error", {{"message", message}, {"type", type}}}};
}
std::string dump(const json& value) {
    return value.dump(-1, ' ', false, json::error_handler_t::replace);
}
void send_error(httplib::Response& response, int status, const std::string& message) {
    response.status = status;
    response.set_content(dump(error_json(message, status == 400 ? "invalid_request_error" :
        (status == 503 ? "server_overloaded" : "server_error"))), "application/json");
}
std::size_t unsigned_field(const json& body, const char* key, std::size_t fallback) {
    if (!body.contains(key)) return fallback;
    const auto& value = body.at(key);
    if (!value.is_number_integer() || (value.is_number_integer() && !value.is_number_unsigned() && value.get<std::int64_t>() < 0))
        throw std::invalid_argument(std::string(key) + " must be a non-negative integer");
    const auto result = value.get<std::uint64_t>();
    if (result > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument(std::string(key) + " is out of range");
    return static_cast<std::size_t>(result);
}
float float_field(const json& body, const char* key, float fallback) {
    if (!body.contains(key)) return fallback;
    if (!body.at(key).is_number()) throw std::invalid_argument(std::string(key) + " must be a number");
    const auto value = body.at(key).get<float>();
    if (!std::isfinite(value)) throw std::invalid_argument(std::string(key) + " must be finite");
    return value;
}
struct Channel {
    GenerationRequest request;
    std::vector<std::pair<std::string, std::string>> messages;
    bool chat = false, stream = false;
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<Event> events;
    bool acknowledged = false;
    int rejection = 0;
    std::string rejection_message;
    std::atomic<bool> cancelled{false};
};
std::shared_ptr<Channel> parse_request(const httplib::Request& request, bool chat,
                                     const std::string& id, const std::string& model) {
    json body;
    try { body = json::parse(request.body); }
    catch (const json::exception&) { throw std::invalid_argument("body must be valid JSON"); }
    if (!body.is_object()) throw std::invalid_argument("body must be a JSON object");
    const std::set<std::string> allowed{"model", chat ? "messages" : "prompt",
        "max_tokens", "temperature", "top_p", "top_k", "seed", "stream"};
    for (auto it = body.begin(); it != body.end(); ++it)
        if (!allowed.count(it.key())) throw std::invalid_argument("unsupported field: " + it.key());
    if (body.contains("model") && !body["model"].is_string())
        throw std::invalid_argument("model must be a string");
    if (body.contains("model") && body["model"].get<std::string>() != model)
        throw std::invalid_argument("model must match the loaded GGUF filename: " + model);
    if (body.contains("stream") && !body["stream"].is_boolean())
        throw std::invalid_argument("stream must be boolean");
    auto channel = std::make_shared<Channel>();
    channel->chat = chat; channel->stream = body.value("stream", false);
    auto& generation = channel->request;
    generation.id = id;
    generation.max_new_tokens = unsigned_field(body, "max_tokens", 32);
    generation.sampling.temperature = float_field(body, "temperature", 0);
    generation.sampling.top_p = float_field(body, "top_p", 1);
    generation.sampling.top_k = unsigned_field(body, "top_k", 0);
    generation.sampling.seed = unsigned_field(body, "seed", 0);
    if (generation.sampling.temperature < 0 || generation.sampling.top_p <= 0 || generation.sampling.top_p > 1)
        throw std::invalid_argument("temperature must be >= 0 and top_p must be in (0, 1]");
    if (chat) {
        if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty())
            throw std::invalid_argument("messages must be a nonempty array");
        for (const auto& message : body["messages"]) {
            if (!message.is_object() || message.size() != 2 || !message.contains("role") ||
                !message.contains("content") || !message["role"].is_string() || !message["content"].is_string())
                throw std::invalid_argument("each message must contain only string role and content");
            const auto role = message["role"].get<std::string>();
            if (role != "system" && role != "user" && role != "assistant")
                throw std::invalid_argument("supported roles: system, user, assistant");
            channel->messages.emplace_back(role, message["content"].get<std::string>());
        }
    } else {
        if (!body.contains("prompt") || !body["prompt"].is_string() || body["prompt"].get_ref<const std::string&>().empty())
            throw std::invalid_argument("prompt must be a nonempty string");
        generation.prompt = body["prompt"].get<std::string>();
    }
    return channel;
}
json usage_json(const Event& event) {
    return {{"prompt_tokens", event.prompt_tokens}, {"completion_tokens", event.completion_tokens},
            {"total_tokens", event.prompt_tokens + event.completion_tokens},
            {"cached_tokens", event.cached_tokens}, {"prefill_tokens", event.prefill_tokens}};
}

class Owner {
public:
    Owner(LlamaEngine& engine, const SchedulerConfig& config)
        : engine_(engine), config_(config), worker_([this](std::stop_token stop) { run(stop); }) {}
    ~Owner() { stop(); }
    void request_stop() {
        { std::lock_guard<std::mutex> lock(mutex_); stopped_ = true; }
        // stop-aware wait 会被 request_stop 唤醒，无须轮询或另配一套停止通知。
        worker_.request_stop();
    }
    void stop() {
        request_stop();
        // 只有 run_server 所在线程 join；watcher 只请求停止，避免并发 join。
        if (worker_.joinable()) worker_.join();
    }
    bool enqueue(const std::shared_ptr<Channel>& channel) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_ || incoming_.size() >= max_pending) return false;
        incoming_.push_back(channel); wake_.notify_one(); return true;
    }
    SchedulerStats stats() {
        std::lock_guard<std::mutex> lock(mutex_); return stats_;
    }
private:
    static void reject(const std::shared_ptr<Channel>& channel, int status, const std::string& message) {
        std::lock_guard<std::mutex> lock(channel->mutex);
        channel->rejection = status; channel->rejection_message = message;
        channel->acknowledged = true; channel->ready.notify_all();
    }
    static void finish_error(const std::shared_ptr<Channel>& channel, const std::string& message) {
        Event end; end.request_id = channel->request.id;
        end.finish_reason = "error"; end.error = message;
        std::lock_guard<std::mutex> lock(channel->mutex);
        channel->rejection = 503; channel->rejection_message = message;
        channel->acknowledged = true;
        // 关闭也服从队列上界，并立即交付终止事件，不再排在未消费 token 后面。
        channel->events.clear(); channel->events.push_back(std::move(end));
        channel->ready.notify_all();
    }
    void run(std::stop_token stop) {
        // Scheduler 在 owner 内构造、销毁：所有 sequence 清理也留在同一个线程。
        std::map<RequestId, std::shared_ptr<Channel>> active;
        std::deque<std::shared_ptr<Channel>> additions;
        const auto fail = [&](const std::string& message) {
            std::cerr << "model owner failed: " << message << '\n';
            for (const auto& channel : additions) reject(channel, 503, message);
            for (const auto& [id, channel] : active) finish_error(channel, message);
            std::lock_guard<std::mutex> lock(mutex_); stats_.healthy = false;
        };
        try {
            Scheduler scheduler(engine_, config_);
            while (!stop.stop_requested()) {
                additions.clear();
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    if (scheduler.idle())
                        wake_.wait(lock, stop, [&] { return !incoming_.empty(); });
                    if (stop.stop_requested()) break;
                    additions.swap(incoming_);
                }
                for (const auto& channel : additions) {
                    if (stop.stop_requested()) { reject(channel, 503, "server is stopping"); continue; }
                    if (channel->cancelled) { reject(channel, 499, "client disconnected"); continue; }
                    try {
                        if (channel->chat) {
                            try { channel->request.prompt = engine_.apply_chat_template(channel->messages, true); }
                            catch (const std::exception& error) { reject(channel, 400, error.what()); continue; }
                        }
                        scheduler.submit(channel->request);
                        active.emplace(channel->request.id, channel);
                        std::lock_guard<std::mutex> lock(channel->mutex);
                        channel->acknowledged = true; channel->ready.notify_all();
                    } catch (const std::invalid_argument& error) { reject(channel, 400, error.what()); }
                      catch (const std::exception& error) { reject(channel, 503, error.what()); }
                }
                if (stop.stop_requested()) break;
                for (const auto& [id, channel] : active)
                    if (channel->cancelled) scheduler.cancel(id);
                if (!scheduler.idle()) {
                    // 停止是协作式的：已经开始的同步 forward 必须返回后才能清理 KV。
                    auto events = scheduler.step();
                    // 先发布本轮统计，再唤醒完成请求：随后 /health 不会看到上一轮快照。
                    { std::lock_guard<std::mutex> lock(mutex_); stats_ = scheduler.stats(); }
                    for (auto& event : events) {
                        const auto found = active.find(event.request_id);
                        if (found == active.end()) continue;
                        const auto channel = found->second;
                        const bool finished = event.kind == EventKind::Finished;
                        {
                            std::lock_guard<std::mutex> lock(channel->mutex);
                            if (channel->events.size() >= max_events) {
                                channel->cancelled = true;
                                channel->events.clear();
                                Event failure; failure.request_id = event.request_id;
                                failure.kind = EventKind::Finished; failure.finish_reason = "error";
                                failure.error = "client consumed events too slowly";
                                channel->events.push_back(std::move(failure));
                            } else if (!channel->cancelled) channel->events.push_back(std::move(event));
                            channel->ready.notify_all();
                        }
                        if (finished) active.erase(found);
                    }
                }
                { std::lock_guard<std::mutex> lock(mutex_); stats_ = scheduler.stats(); }
            }
            for (auto& [id, channel] : active) {
                scheduler.cancel(id);
                finish_error(channel, "server is stopping");
            }
            // 所有 live request 已标记 cancel；step 只回收，不再发起模型前向。
            while (!scheduler.idle()) scheduler.step();
        } catch (const std::exception& error) {
            // 即使 scheduler 抛出非预期异常，所有等 ack/终止事件的 HTTP 线程也必须醒来。
            fail(error.what());
        } catch (...) {
            fail("unknown model owner failure");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        for (const auto& channel : incoming_) reject(channel, 503, "server is stopping");
        incoming_.clear();
    }
    LlamaEngine& engine_;
    SchedulerConfig config_;
    std::mutex mutex_;
    std::condition_variable_any wake_;
    std::deque<std::shared_ptr<Channel>> incoming_;
    SchedulerStats stats_;
    bool stopped_ = false; // mutex_ 保护入口是否仍接受请求。
    // 最后构造、最先析构，join 完成前以上状态与 engine_ 引用都保持有效。
    std::jthread worker_;
};

bool acknowledge(const httplib::Request& request, const std::shared_ptr<Channel>& channel) {
    std::unique_lock<std::mutex> lock(channel->mutex);
    while (!channel->acknowledged) {
        channel->ready.wait_for(lock, 50ms);
        if (request.is_connection_closed && request.is_connection_closed()) {
            channel->cancelled = true; return false;
        }
    }
    return true;
}
bool next_event(const std::shared_ptr<Channel>& channel, Event& event) {
    std::unique_lock<std::mutex> lock(channel->mutex);
    channel->ready.wait_for(lock, 50ms, [&] { return !channel->events.empty(); });
    if (channel->events.empty()) return false;
    event = std::move(channel->events.front()); channel->events.pop_front(); return true;
}
json response_base(const Channel& channel, const ServerConfig& config, bool stream) {
    const auto created = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    return {{"id", channel.request.id}, {"object", channel.chat ?
        (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion"},
        {"created", created}, {"model", config.model_name}};
}
void nonstream_response(const httplib::Request& request, httplib::Response& response,
                        const std::shared_ptr<Channel>& channel, const ServerConfig& config) {
    std::string text;
    for (;;) {
        if (request.is_connection_closed && request.is_connection_closed()) { channel->cancelled = true; return; }
        Event event;
        if (!next_event(channel, event)) continue;
        if (event.kind == EventKind::Token) { text += event.text; continue; }
        if (!event.error.empty()) { send_error(response, 500, event.error); return; }
        auto output = response_base(*channel, config, false);
        const std::string cleaned = consume_utf8(text, true);
        json choice = {{"index", 0}, {"finish_reason", event.finish_reason}};
        if (channel->chat) choice["message"] = {{"role", "assistant"}, {"content", cleaned}};
        else choice["text"] = cleaned;
        output["choices"] = json::array({choice}); output["usage"] = usage_json(event);
        response.set_content(dump(output), "application/json"); return;
    }
}
void streaming_response(httplib::Response& response, const std::shared_ptr<Channel>& channel,
                        const ServerConfig& config) {
    struct StreamState { std::string pending; bool sent_role = false; };
    auto state = std::make_shared<StreamState>();
    const auto base = response_base(*channel, config, true);
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
    response.set_chunked_content_provider("text/event-stream",
        [channel, state, base](std::size_t, httplib::DataSink& sink) {
            auto write = [&](const json& value) {
                const auto data = "data: " + dump(value) + "\n\n";
                return sink.write(data.data(), data.size());
            };
            if (channel->chat && !state->sent_role) {
                auto output = base;
                output["choices"] = json::array({{{"index", 0}, {"delta", {{"role", "assistant"}}}, {"finish_reason", nullptr}}});
                state->sent_role = true;
                if (!write(output)) { channel->cancelled = true; return false; }
            }
            for (;;) {
                if (sink.is_writable && !sink.is_writable()) { channel->cancelled = true; return false; }
                Event event;
                if (!next_event(channel, event)) continue;
                const bool final = event.kind == EventKind::Finished;
                state->pending += event.text;
                const auto content = consume_utf8(state->pending, final);
                if (!content.empty()) {
                    auto output = base;
                    json choice = {{"index", 0}, {"finish_reason", nullptr}};
                    if (channel->chat) choice["delta"] = {{"content", content}}; else choice["text"] = content;
                    output["choices"] = json::array({choice});
                    if (!write(output)) { channel->cancelled = true; return false; }
                }
                if (final) {
                    if (!event.error.empty()) {
                        if (!write(error_json(event.error, "server_error"))) { channel->cancelled = true; return false; }
                    } else {
                        auto output = base;
                        json choice = {{"index", 0}, {"finish_reason", event.finish_reason}};
                        if (channel->chat) choice["delta"] = json::object(); else choice["text"] = "";
                        output["choices"] = json::array({choice}); output["usage"] = usage_json(event);
                        if (!write(output)) { channel->cancelled = true; return false; }
                    }
                    const std::string done = "data: [DONE]\n\n";
                    if (!sink.write(done.data(), done.size())) { channel->cancelled = true; return false; }
                    sink.done(); return true;
                }
                // 一次 provider 调用交付一轮，交还 httplib 检查 socket 状态。
                return true;
            }
        }, [channel](bool) { channel->cancelled = true; });
}
} // namespace

int run_server(LlamaEngine& engine, const SchedulerConfig& scheduler, const ServerConfig& config) {
    SignalHandlers signals;
    Owner owner(engine, scheduler);
    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(8, 32); };
    server.set_payload_max_length(1024 * 1024);
    server.set_read_timeout(10, 0);
    server.set_write_timeout(10, 0);
    server.set_keep_alive_max_count(32);
    server.set_keep_alive_timeout(5);
    std::atomic<std::uint64_t> sequence{0};
    server.Get("/health", [&](const httplib::Request&, httplib::Response& response) {
        const auto s = owner.stats();
        response.status = s.healthy ? 200 : 503;
        response.set_content(dump({{"status", s.healthy ? "ok" : "error"},
            {"stats", {{"active_requests", s.active_requests}, {"queued_requests", s.queued_requests},
                {"cache_entries", s.cache_entries}, {"cache_tokens", s.cache_tokens},
                {"reserved_tokens", s.reserved_tokens}, {"free_sequences", s.free_sequences},
                {"total_submitted", s.total_submitted}, {"total_finished", s.total_finished},
                {"total_cancelled", s.total_cancelled}, {"total_failed", s.total_failed},
                {"prefill_tokens", s.prefill_tokens}, {"decode_tokens", s.decode_tokens},
                {"cache_hit_tokens", s.cache_hit_tokens}, {"forward_calls", s.forward_calls}}}}), "application/json");
    });
    const auto handler = [&](bool chat) {
        return [&, chat](const httplib::Request& request, httplib::Response& response) {
            try {
                const auto channel = parse_request(request, chat, "cmpl-" + std::to_string(sequence++), config.model_name);
                if (!owner.enqueue(channel)) { send_error(response, 503, "pending request queue is full"); return; }
                if (!acknowledge(request, channel)) return;
                {
                    std::lock_guard<std::mutex> lock(channel->mutex);
                    if (channel->rejection) { send_error(response, channel->rejection, channel->rejection_message); return; }
                }
                if (channel->stream) streaming_response(response, channel, config);
                else nonstream_response(request, response, channel, config);
            } catch (const std::invalid_argument& error) { send_error(response, 400, error.what()); }
              catch (const json::exception& error) { send_error(response, 400, error.what()); }
              catch (const std::exception& error) { send_error(response, 500, error.what()); }
        };
    };
    server.Post("/v1/completions", handler(false));
    server.Post("/v1/chat/completions", handler(true));
    // 等待设施先在调用线程构造；watcher 启动异常时仍能正常析构 Owner。
    std::mutex signal_mutex;
    std::condition_variable_any signal_wake;
    std::jthread signal_watcher([&](std::stop_token stop) {
        std::unique_lock<std::mutex> lock(signal_mutex);
        while (!stop.stop_requested()) {
            if (interrupted) {
                owner.request_stop();
                // bind 后、listen 前到达的信号不能丢：httplib 此时 stop 尚无效。
                if (server.is_running()) { server.stop(); return; }
            }
            // 信号处理器仅设置 sig_atomic_t；普通线程负责停止和 HTTP 操作。
            signal_wake.wait_for(lock, stop, 50ms, [] { return false; });
        }
    });
    if (!server.bind_to_port(config.host, config.port)) {
        throw std::runtime_error("cannot bind HTTP server to " + config.host + ":" + std::to_string(config.port));
    }
    std::cerr << "mini-sglang listening on http://" << config.host << ':' << config.port << '\n';
    const bool ok = server.listen_after_bind();
    signal_watcher.request_stop(); signal_watcher.join();
    owner.stop();
    return ok ? 0 : 1;
}
} // namespace minisgl
