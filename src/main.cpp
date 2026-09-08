// CLI 是组合根：在这里配置真实模型、预算和输出，核心不依赖终端或 JSON。
#include "minisgl/llama_engine.hpp"
#include "minisgl/scheduler.hpp"
#include "minisgl/server.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using nlohmann::json;
struct Options {
    minisgl::LlamaConfig model;
    minisgl::SchedulerConfig scheduler;
    minisgl::ServerConfig server;
    minisgl::SamplingParams sampling;
    std::vector<std::string> prompts;
    std::size_t max_tokens = 32, active = 8, cache = 8;
    bool serve = false, json_output = false, help = false;
    Options() { model.context_tokens = 4096; model.batch_size = 256; }
};

std::uint64_t integer(const std::string& value, const std::string& flag) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(flag + " must be a non-negative integer");
    std::size_t used = 0;
    auto number = std::stoull(value, &used);
    if (used != value.size()) throw std::invalid_argument(flag + " is invalid");
    return number;
}
template<class T> T narrow(std::uint64_t value, const std::string& flag) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<T>::max()))
        throw std::invalid_argument(flag + " is out of range");
    return static_cast<T>(value);
}
float number(const std::string& value, const std::string& flag) {
    std::size_t used = 0;
    auto result = std::stof(value, &used);
    if (used != value.size() || !std::isfinite(result))
        throw std::invalid_argument(flag + " must be finite");
    return result;
}
void usage() {
    std::cout << R"(mini-sglang: native C++ scheduling with a real llama.cpp GGUF backend
Usage:
  mini-sglang --model model.gguf --prompt TEXT [--prompt TEXT ...] [--json]
  mini-sglang --model model.gguf --serve [--host 127.0.0.1 --port 1919]

Generation: --max-tokens 32 --temperature 0 --top-p 1 --top-k 0 --seed 0
Scheduling: --batch-tokens 256 --prefill-chunk 64 --max-sequences 8
            --context-size 4096 --cache-sequences 8
Backend:    --threads 4 --gpu-layers 0

--context-size is the total KV token budget, shared by active requests/cache.
--max-sequences limits active requests; cache sequences use additional slots.
--max-tokens 0 validates/tokenizes the request without a model forward pass.
Chat requires a chat template stored in the GGUF model. No model is bundled.
)";
}
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") { o.help = true; continue; }
        if (flag == "--serve") { o.serve = true; continue; }
        if (flag == "--json") { o.json_output = true; continue; }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + flag);
        const std::string value = argv[++i];
        if (flag == "--model") o.model.model_path = value;
        else if (flag == "--prompt") o.prompts.push_back(value);
        else if (flag == "--host") o.server.host = value;
        else if (flag == "--port") o.server.port = narrow<std::uint16_t>(integer(value, flag), flag);
        else if (flag == "--max-tokens") o.max_tokens = narrow<std::size_t>(integer(value, flag), flag);
        else if (flag == "--temperature") o.sampling.temperature = number(value, flag);
        else if (flag == "--top-p") o.sampling.top_p = number(value, flag);
        else if (flag == "--top-k") o.sampling.top_k = narrow<std::size_t>(integer(value, flag), flag);
        else if (flag == "--seed") o.sampling.seed = integer(value, flag);
        else if (flag == "--batch-tokens") o.model.batch_size = narrow<std::uint32_t>(integer(value, flag), flag);
        else if (flag == "--prefill-chunk") o.scheduler.prefill_chunk_size = narrow<std::size_t>(integer(value, flag), flag);
        else if (flag == "--max-sequences") o.active = narrow<std::size_t>(integer(value, flag), flag);
        else if (flag == "--context-size") o.model.context_tokens = narrow<std::uint32_t>(integer(value, flag), flag);
        else if (flag == "--cache-sequences") o.cache = narrow<std::size_t>(integer(value, flag), flag);
        else if (flag == "--threads") o.model.threads = narrow<std::int32_t>(integer(value, flag), flag);
        else if (flag == "--gpu-layers") o.model.gpu_layers = narrow<std::int32_t>(integer(value, flag), flag);
        else throw std::invalid_argument("unknown option: " + flag);
    }
    if (o.help) return o;
    if (o.model.model_path.empty()) throw std::invalid_argument("--model is required (a real GGUF file)");
    if (o.serve && (!o.prompts.empty() || o.json_output))
        throw std::invalid_argument("--serve cannot be combined with --prompt or --json");
    if (!o.serve && o.prompts.empty()) throw std::invalid_argument("provide --prompt or --serve");
    if (o.server.host.empty() || o.server.port == 0 || o.model.threads <= 0)
        throw std::invalid_argument("host must be nonempty; port and threads must be positive");
    if (o.sampling.temperature < 0 || o.sampling.top_p <= 0 || o.sampling.top_p > 1)
        throw std::invalid_argument("temperature must be >= 0 and top-p must be in (0, 1]");
    if (o.cache > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        o.active > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) - o.cache)
        throw std::invalid_argument("sequence count is too large");
    o.model.max_sequences = narrow<std::uint32_t>(o.active + o.cache, "sequence count");
    o.scheduler.max_running_requests = o.active;
    o.scheduler.max_pending_requests = 128;
    o.scheduler.max_batch_tokens = o.model.batch_size;
    o.scheduler.max_sequence_tokens = o.model.context_tokens;
    o.scheduler.max_total_tokens = o.model.context_tokens;
    o.scheduler.max_sequences = o.model.max_sequences;
    o.scheduler.max_cache_entries = o.cache;
    o.scheduler.max_cache_tokens = o.cache ? o.model.context_tokens / 2 : 0;
    minisgl::validate_config(o.scheduler);
    o.server.model_name = std::filesystem::path(o.model.model_path).filename().string();
    return o;
}
json stats_json(const minisgl::SchedulerStats& s) {
    return {{"active_requests", s.active_requests}, {"queued_requests", s.queued_requests},
            {"cache_entries", s.cache_entries}, {"cache_tokens", s.cache_tokens},
            {"reserved_tokens", s.reserved_tokens}, {"free_sequences", s.free_sequences},
            {"total_submitted", s.total_submitted}, {"total_finished", s.total_finished},
            {"total_cancelled", s.total_cancelled}, {"total_failed", s.total_failed},
            {"prefill_tokens", s.prefill_tokens}, {"decode_tokens", s.decode_tokens},
            {"cache_hit_tokens", s.cache_hit_tokens}, {"forward_calls", s.forward_calls},
            {"healthy", s.healthy}};
}
int offline(minisgl::LlamaEngine& engine, const Options& o) {
    minisgl::Scheduler scheduler(engine, o.scheduler);
    struct Result { std::string text; std::vector<minisgl::Token> tokens; minisgl::Event end; };
    std::vector<Result> results(o.prompts.size());
    std::map<minisgl::RequestId, std::size_t> indices;
    std::size_t next = 0, finished = 0;
    bool failed = false;
    while (finished < results.size()) {
        // 输入规模不限于 pending queue：按队列容量逐批提交，仍按原输入顺序输出。
        while (next < o.prompts.size() && scheduler.stats().queued_requests < o.scheduler.max_pending_requests) {
            minisgl::GenerationRequest request;
            request.id = std::to_string(next); request.prompt = o.prompts[next];
            request.max_new_tokens = o.max_tokens; request.sampling = o.sampling;
            scheduler.submit(request);
            indices.emplace(request.id, next++);
        }
        for (const auto& event : scheduler.step()) {
            auto& result = results.at(indices.at(event.request_id));
            if (event.kind == minisgl::EventKind::Token) {
                result.text += event.text; result.tokens.push_back(event.token);
            } else {
                result.end = event; ++finished;
                if (!event.error.empty()) failed = true;
            }
        }
    }
    json output = {{"results", json::array()}, {"stats", stats_json(scheduler.stats())}};
    for (std::size_t i = 0; i < results.size(); ++i) {
        auto& r = results[i];
        std::string text = minisgl::consume_utf8(r.text, true);
        json item = {{"request_id", std::to_string(i)}, {"prompt", o.prompts[i]},
            {"text", text}, {"token_ids", r.tokens}, {"finish_reason", r.end.finish_reason},
            {"usage", {{"prompt_tokens", r.end.prompt_tokens}, {"completion_tokens", r.end.completion_tokens},
             {"cached_tokens", r.end.cached_tokens}, {"prefill_tokens", r.end.prefill_tokens}}}};
        if (!r.end.error.empty()) item["error"] = r.end.error;
        output["results"].push_back(item);
        if (!o.json_output) {
            if (results.size() > 1) std::cout << "[" << i << "] ";
            std::cout << text << '\n';
            if (!r.end.error.empty()) std::cerr << "request " << i << ": " << r.end.error << '\n';
        }
    }
    if (o.json_output) std::cout << output.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
    return failed ? 1 : 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (options.help) { usage(); return 0; }
        minisgl::LlamaEngine engine(options.model);
        return options.serve ? minisgl::run_server(engine, options.scheduler, options.server)
                             : offline(engine, options);
    } catch (const std::exception& error) {
        std::cerr << "mini-sglang: " << error.what() << '\n';
        return 1;
    }
}
