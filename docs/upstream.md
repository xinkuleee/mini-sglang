# 与官方实现的关系

Python main 完整保留 [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang) 的提交 9a91cfafe754aa85daee49998176275667eb58f2。cpp 与 rust 从这个提交分叉，保留 MIT 许可证。

两个原生版本重建请求调度、采样、前缀缓存、生命周期和服务协议。真实模型计算采用固定版本 [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/64e9bceb2c3a856efed96feda784a50947049feb) 的薄 C ABI 适配层，负责 GGUF 权重、tokenizer、张量运算和物理 KV。没有调用 Python，没有转发到 llama-server，也没有用固定答案替代模型前向。

| 官方模块/机制 | 原生版本的处理 | 原因 |
|---|---|---|
| core.Req/Batch | 显式请求状态和 token/position/sequence batch | 区分已计算 KV 和刚采样的 token，防止位置错误 |
| scheduler | 单一所有者、动态批处理、有界准入 | 多线程只处理连接，模型状态无需细粒度锁 |
| Chunked prefill | 每轮有限 token 预算，长输入分块 | decode 优先，剩余预算给 prefill，限制长输入对现有请求的干扰 |
| Radix cache | 原生压缩 radix 索引、LRU、真实 sequence KV 复用 | trie 命中必须实际减少送入模型的 token |
| MHAKVCache 物理页 | 交给 llama.cpp memory API | 不引入无法控制真实显存的影子页分配器 |
| engine.sample | 原生 temperature/top-k/top-p 与请求级随机状态 | 请求的随机状态不受其他请求调度顺序影响 |
| PyTorch 模型与自定义 CUDA kernel | llama.cpp GGUF backend | 保持可移植的真实计算路径，阅读重点放在服务运行时 |
| Tokenizer/detokenizer 进程 | 后端 tokenizer 与前端 UTF-8 分片处理 | 小型单进程服务无需 ZMQ 和重复模型元数据 |
| FastAPI | 各语言原生 HTTP、JSON、SSE | 网络协议不进入 scheduler 和模型层 |
| CUDA Graph / 双流 overlap | 未移植 | 主机线程并发不等于 GPU overlap scheduling |
| TP / NCCL | 未移植 | 不实现多 GPU rank、权重分片或 collective |
| FlashAttention / FlashInfer / TRTLLM | 未移植官方集成 | 后端优化不能冒充官方 kernel 的移植 |

因此，这两个分支是**参考官方机制的原生教学推理运行时**，不是官方全特性、性能等价的机械翻译。GGUF 与官方 Hugging Face/PyTorch 权重入口不同。GPU 可用性与模型支持范围取决于构建的 llama.cpp backend；实测范围见 [validation.md](validation.md)。

## 有意做出的简化

1. 官方默认先调度 prefill；这里先给每个正在 decode 的请求一个位置，再用剩余预算处理 prefill。活动请求数受 batch 预算限制，新请求不夺走现有 decode 的位置。
2. 官方中间 prefill 块的采样结果会被丢弃；这里中间块不请求 logits、不采样，减少无用计算，也避免消耗随机数。
3. 缓存只保存完整 prompt 的已计算 KV，不缓存任意生成历史。索引仍支持部分公共前缀命中；这是一个容易验证的边界。
4. 命中最多覆盖 prompt_length - 1 个 token；最后一个 token 必须重算以获得本请求下一 token 的 logits。
5. 活动序列与缓存序列使用同一个有界 ID 池；逻辑 KV 预算保守计费，即使后端物理共享也不超售。没有推测式调度、抢占恢复、模型热切换。
6. 对未支持的 API 参数明确报错，不静默接受后忽略。不支持图片、工具调用、任意 Jinja 模板或生产级认证。

## 阅读官方时的定位

- 请求长度不变量：[python/minisgl/core.py](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/core.py)。
- 准入、分块与 decode：[python/minisgl/scheduler](https://github.com/sgl-project/mini-sglang/tree/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler)。
- radix 树与引用保护：[radix_cache.py](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/kvcache/radix_cache.py)。
- CUDA 执行边界：[engine.py](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/engine/engine.py)。
