# Mini-SGLang · C++

参考官方 mini-sglang 的 C++20 教学推理运行时：原生调度、压缩 radix 前缀缓存、分块 prefill、连续批处理和请求级采样，使用 llama.cpp 执行真实 GGUF 模型。HTTP 支持文本补全、基于模型模板的聊天补全和 SSE 流式输出。

| 分支 | 内容 |
|---|---|
| [main](https://github.com/xinkuleee/mini-sglang/tree/main) | 官方 Python 源码与历史 |
| [cpp](https://github.com/xinkuleee/mini-sglang/tree/cpp) | 本版本 |
| [rust](https://github.com/xinkuleee/mini-sglang/tree/rust) | Rust 调度、缓存、采样与服务 |

**实现边界：** 两个原生分支复现推理服务的核心机制，计算层使用固定版本 llama.cpp，不是官方 CUDA kernel、GPU overlap、TP/NCCL 的完整移植。[上游映射与差异](docs/upstream.md)列出已实现与未移植内容。

## 构建与运行

需要支持 C++20 的编译器和标准库（包含 std::span、std::jthread）、CMake 3.24+、Git。默认只构建核心库与语义测试，不下载模型或 llama.cpp：

```bash
cmake -S . -B build-core -DCMAKE_BUILD_TYPE=Release
cmake --build build-core --config Release --parallel
ctest --test-dir build-core -C Release --output-on-failure
```

构建真实推理 CLI 和 HTTP 服务（首次配置下载固定版本依赖）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMINISGL_WITH_LLAMA=ON
cmake --build build --config Release --parallel
./build/mini-sglang --model /path/model.gguf --prompt "Once upon a time" --max-tokens 32
./build/mini-sglang --model /path/model.gguf --prompt "Hello" --prompt "Good morning" --json
./build/mini-sglang --model /path/model.gguf --serve --host 127.0.0.1 --port 1919
```

Windows Visual Studio 多配置构建的程序位于 build/Release/mini-sglang.exe；Ninja 构建位于 build/mini-sglang.exe。执行 --help 查看参数。使用已有依赖时，设置 MINISGL_LLAMA_SOURCE、MINISGL_HTTPLIB_SOURCE 和 MINISGL_JSON_SOURCE；llama.cpp checkout 必须匹配固定提交。

CUDA 为可选构建选项 -DMINISGL_CUDA=ON，运行时 --gpu-layers 指定卸载层数。本机只验证 CPU 路径，不将该选项视为官方 GPU 性能等价证明。

主运行时使用 C++20；native/ 中的薄 C ABI 桥与固定版本 llama.cpp 仍按 C++17 编译，两者通过 C ABI 交互。

## 学习入口

- [架构与模块职责](docs/architecture.md)：每个模块做什么、边界在哪里、为什么这样设计。
- [官方实现映射](docs/upstream.md)：原版机制、简化与未移植能力。
- [验证与复现实验](docs/validation.md)：测试命令、模型、实测范围。
- [HTTP 与资源参数](docs/usage.md)：请求格式、SSE、容量和限制。

建议依次读 types → engine → sampling → cache → scheduler → native bridge → server。生产推理不依赖 Python；tests 中的 Python 脚本仅作为黑盒验收客户端。
