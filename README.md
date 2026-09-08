# Mini-SGLang · Rust

参考官方 mini-sglang 的 Rust 教学推理运行时：原生调度、压缩 radix 前缀缓存、分块 prefill、连续批处理和请求级采样，使用 llama.cpp 执行真实 GGUF 模型。HTTP 支持文本补全、基于模型模板的聊天补全和 SSE 流式输出。

| 分支 | 内容 |
|---|---|
| [main](https://github.com/xinkuleee/mini-sglang/tree/main) | 官方 Python 源码与历史 |
| [cpp](https://github.com/xinkuleee/mini-sglang/tree/cpp) | C++ 调度、缓存、采样与服务 |
| [rust](https://github.com/xinkuleee/mini-sglang/tree/rust) | 本版本 |

**实现边界：** 两个原生分支复现推理服务的核心机制，计算层使用固定版本 llama.cpp，不是官方 CUDA kernel、GPU overlap、TP/NCCL 的完整移植。[上游映射与差异](docs/upstream.md)列出已实现与未移植内容。

## 构建与运行

需要 Rust/Cargo；启用真实模型还需要 C++17 编译器、CMake 3.24+ 和 Git。核心逻辑是 Rust，只有底层模型计算桥使用 C++。默认不启用 llama feature：

```bash
cargo test --locked
cargo run -- --help
```

构建真实推理 CLI 和 HTTP 服务（首次构建下载固定版本 llama.cpp）：

```bash
cargo build --release --locked --features llama
./target/release/mini-sglang --model /path/model.gguf --prompt "Once upon a time" --max-tokens 32
./target/release/mini-sglang --model /path/model.gguf --prompt "Hello" --prompt "Good morning" --json
./target/release/mini-sglang --model /path/model.gguf --serve --host 127.0.0.1 --port 1919
```

设置环境变量 MINISGL_LLAMA_SOURCE 可以使用已检出的固定提交；MINISGL_NATIVE_JOBS 控制 C++ 并行构建数。Windows GNU 工具链需要配套 MinGW/Clang 与 Ninja；MSVC 工具链需要 Visual Studio Build Tools。运行时携带 build.rs 生成的 minisgl_bridge 动态库，Windows 已复制到程序相邻目录；Linux/macOS 发行时一并分发该库。

MINISGL_CUDA=ON 启用可选 CUDA 构建，运行时 --gpu-layers 指定卸载层数。本机仅验证 CPU，未测试 CUDA。执行 --help 查看参数。

## 学习入口

- [架构与模块职责](docs/architecture.md)：每个模块做什么、边界在哪里、为什么这样设计。
- [官方实现映射](docs/upstream.md)：原版机制、简化与未移植能力。
- [验证与复现实验](docs/validation.md)：测试命令、模型、实测范围。
- [HTTP 与资源参数](docs/usage.md)：请求格式、SSE、容量和限制。

建议依次读 types → engine → sampling → cache → scheduler → native bridge → server。生产推理不依赖 Python；tests 中的 Python 脚本仅作为黑盒验收客户端。
