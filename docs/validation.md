# 可复现验证

核心测试不需要 GPU、Python 或模型权重。测试用引擎只存在于 tests，负责故障注入和确定性算法验证；真实 GGUF 集成测试另行执行。

## 真实模型黑盒检查

下载一个公开的小模型（约 1.2 MB），不要将权重提交到仓库：

```bash
mkdir -p .models
curl -L https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf -o .models/stories260K.gguf
```

模型 SHA256：270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d。它用于验证计算正确性，不用于评价语言质量。带聊天模板的补充测试使用 SmolLM2-135M-Instruct-Q4_K_M.gguf，SHA256：2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d。

```bash
python tests/conformance.py --binary PATH_TO_MINI_SGLANG --model .models/stories260K.gguf
python tests/conformance.py --cpp CPP_BINARY --rust RUST_BINARY --model .models/stories260K.gguf
```

脚本比较真实 token IDs：单请求与动态批处理、prefill 块大小 1/7/64、完整和部分前缀命中、缓存淘汰后的重算。它还验证 max_tokens=0 不做前向、无效采样参数被拒绝。缓存收益检查实际发给 backend 的 prefill token 数，相同 prompt 的第二次请求应只重算最后一个 token。

两种语言只要求 greedy 输出在相同模型、后端和构建下相同。随机采样的 RNG 算法不同，不承诺同一个 seed 跨语言得到相同序列。每种语言内的请求级确定性由核心测试覆盖。CPU/GPU 数值路径也可能改变接近相等 logits 的 argmax；不承诺跨设备逐位一致。

## 实测环境与范围

2026-09-08，Windows x86_64 / Intel CPU，无 NVIDIA GPU；Clang 23.1.0（llvm-mingw 20260826）、CMake 4.4.3、Ninja 1.13.2、Rust 1.98.1。工具链安装在工作区，未修改全局环境。

以下为本地实测结果；CI 工作流也已提供，远端 CI 是否运行另行说明。

| 检查 | 结果 |
|---|---|
| 核心调度、缓存、采样语义 | 10 组核心语义测试通过；包含 3,000 次固定种子 radix oracle，CTest 1/1 |
| 真实 GGUF C ABI 前向、KV 复制/回收 | 通过；tiny Llama 与 SmolLM2，复制源释放后前缀仍可用，与完整 replay logits 一致 |
| CLI 模型、batch、chunk、prefix 一致性 | 通过；两语言 greedy token IDs 一致，chunk 1/7/64、完整/部分命中、淘汰重算、零输出均通过 |
| HTTP / chat / SSE / 错误请求 | 通过；真实 SmolLM2 模板、流与非流输出一致、6 并发、未知字段/非法参数/缺少模板拒绝 |
| CUDA、TP、上游 Python 数值/性能等价 | 未验证；TP 与官方 CUDA 集成未移植 |

原版 Python main 的 CUDA 环境在本机不存在，因此没有声称已运行原版 GPU 测试，也没有给出与上游吞吐性能的比较。

真实缓存收益：tiny 测试相同 prompt 的 prefill 从 10 降到 1 个 token；SmolLM2 HTTP 从 28 降到 1，复用 27 个 token。这里记录的是实际前向输入数量，不是估算的加速倍数。

最终 SmolLM2 流式断连测试启用 --require-cancel，实际观测 total_cancelled 增加 1，活动/排队/预留资源全部归零。

复现 HTTP 验证：

```bash
python tests/server_smoke.py --binary PATH_TO_MINI_SGLANG --model /path/SmolLM2-135M-Instruct-Q4_K_M.gguf --chat --require-cancel
python tests/server_smoke.py --binary PATH_TO_MINI_SGLANG --model .models/stories260K.gguf --expect-no-chat
```
