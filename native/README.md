# 计算桥：职责与验证

这里的 C ABI 由 C++ 和 Rust 分支保留相同实现。它只负责 GGUF 模型生命周期、分词、原始 token 字节、模型 chat template、batch forward、logits 拷出和 KV 序列引用。请求队列、chunked prefill、continuous batching、radix 前缀索引、内存预算、采样和流式响应属于各语言自己的模块。

复用 llama.cpp 内核使教学实现集中在推理服务运行时；这不是逐行移植 Python 模型，也不宣称覆盖上游 CUDA/Triton kernels、张量并行、分布式推理或全部模型。固定计算依赖为 `ggml-org/llama.cpp@64e9bceb2c3a856efed96feda784a50947049feb`。本地 checkout 会校验 HEAD，不接受任意版本的 ABI。

| 文件 | 职责 | 这样设计的原因 |
| --- | --- | --- |
| `minisgl_bridge.h` | 固定宽度 C ABI，调用者分配输出缓冲区 | 避免让 Rust 依赖 C++ STL、异常或不同 allocator |
| `minisgl_bridge.cpp` | 校验输入，持有 model/context，将 batch 转成 llama 输入 | 模型计算与调度策略分离；所有 C++ 异常在边界内变成错误 |
| `CMakeLists.txt` | 固定依赖，默认 CPU，可选 CUDA | 相同内核版本用于两语言比较；核心测试无需下载依赖 |
| `abi_smoke.cpp` | 错误边界与真实模型不变量验证 | 用真实 logits 验证 KV 共享的语义，而非假引擎输出 |

`context_tokens` 是所有请求共享的总 KV token 池。`kv_unified=true` 使单序列上限也等于该容量，而不会按 `max_sequences` 切成固定分区。`swa_full=true` 避免后端自动丢掉仍被 radix 索引持有的前缀。序列编号必须在 `[0,max_sequences)`；scheduler 保证每个序列的位置从 0 连续追加。复制前缀的目标必须为空，范围是 `[0,end_exclusive)`。复制只增加 KV cell 的序列引用，删除源引用不会删除目标仍引用的 KV。

不支持 recurrent、hybrid、encoder-decoder 或 diffusion 模型：它们的状态复制/注意力语义不满足这里的共享前缀契约。GGUF 必须带可被 `llama_chat_apply_template` 支持的模板才能调用 chat API；该 API 不是通用 Jinja 解释器。缺少或不支持模板会返回明确错误，文本 completion 仍可用于支持的模型。token piece 可能只有部分 UTF-8 字节，调用者应先积累字节再输出完整字符。

所有 `msgl_*` 操作均不抛出异常跨越 C 边界。`-1` 是错误，`msgl_error()` 返回线程局部错误消息；可变长输出的 `1` 表示需要按 `required` 扩容，`0` 是成功。`is_eog` 的成功值为 0/1，`vocab_size` 返回正数。一个 handle 由一个 scheduler 线程独占。Rust wrapper 的 `Drop` 与 C++ wrapper 的 RAII 释放 context 和 model；backend 注册表按进程初始化，避免一个实例析构破坏另一个实例。

## 单独编译和真实模型 smoke

需要 CMake 3.24+ 和 C++17 编译器。以下命令在当前分支根目录运行，模型路径指向已下载且许可允许使用的 GGUF。省略 `MINISGL_LLAMA_SOURCE` 时 CMake 从 GitHub 获取固定 commit。

```sh
cmake -S native -B build-native -DCMAKE_BUILD_TYPE=Release \
  -DMINISGL_NATIVE_TESTS=ON -DMINISGL_LLAMA_SOURCE=/path/to/llama.cpp
cmake --build build-native --config Release --parallel 4
ctest --test-dir build-native --output-on-failure -C Release
./build-native/lib/minisgl-abi-smoke /path/to/model.gguf
```

Windows 使用 Ninja 或本机 Visual Studio generator；Ninja 单配置输出为 `build-native/lib/minisgl-abi-smoke.exe`。smoke 验证非法参数错误、真实分词、prefill logits、共享前缀在源序列删除后的可用性、共享 decode 与未缓存 replay 的 logits 一致性、多序列交错 batch 的输出行顺序以及 chat template。

CPU 默认 `GGML_NATIVE=OFF`、`MINISGL_CPU_BASELINE=ON`、CUDA/OpenMP 关闭。baseline 显式关闭 AVX/AVX2/FMA/F16C/BMI2/SSE4.2，便于在较旧的 x86-64 CPU 运行；这不是最快的 CPU 内核配置。支持 AVX2 的机器可用 `-DMINISGL_CPU_BASELINE=OFF` 开启 llama 的默认 x86 优化。可用 `-DMINISGL_CUDA=ON` 构建 CUDA，需匹配的 CUDA Toolkit、驱动和 cuBLAS/cuDART 运行库。GPU 路径尚未在这里的 CPU 验证环境中测试。

## Rust 动态库

Rust 的 `build.rs` 只在 `--features llama` 时调用 CMake，将静态 llama/ggml 链入单个 adapter 动态库，避免手工枚举传递链接参数。不使用额外 Cargo build dependency。`MINISGL_LLAMA_SOURCE`、`MINISGL_CUDA=ON`、`MINISGL_CPU_BASELINE=OFF` 和 `MINISGL_NATIVE_JOBS` 可控制构建。Windows 将 adapter DLL 放在 `target/{profile}` 和 `deps` 供运行/测试；Linux/macOS 设置开发 rpath 并保留可执行文件相邻库的路径。发行时携带 adapter 动态库和编译器/可选 GPU runtime 所需库。暂不支持 Rust 交叉编译。
