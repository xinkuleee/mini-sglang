#pragma once

// 传输层只负责参数、HTTP 和字节编码；模型/KV/采样的生命周期属于 Scheduler。
#include "minisgl/types.hpp"
#include <cstdint>
#include <string>

namespace minisgl {
class LlamaEngine;
struct ServerConfig {
    std::string host = "127.0.0.1";
    std::uint16_t port = 1919;
    std::string model_name;
};

// token 可以把一个 UTF-8 字符拆开。仅交付完整字符，结束时替换残缺字节。
std::string consume_utf8(std::string& pending, bool finish = false);
int run_server(LlamaEngine& engine, const SchedulerConfig& scheduler,
               const ServerConfig& server);
} // namespace minisgl
