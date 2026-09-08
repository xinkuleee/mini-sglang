# 使用与容量

## CLI

```bash
mini-sglang --model model.gguf --prompt "Hello" --max-tokens 32 --temperature 0
mini-sglang --model model.gguf --prompt "One" --prompt "Two" --json
mini-sglang --model model.gguf --serve --port 1919
```

| 参数 | 默认值 | 含义 |
|---|---:|---|
| --context-size | 4096 | 活动请求与缓存共享的总 KV token 预算 |
| --max-sequences | 8 | 同时活动的请求数；不得超过 batch token 预算 |
| --cache-sequences | 8 | 独立缓存 sequence 数；0 关闭缓存 |
| --batch-tokens | 256 | 每次模型 forward 最多处理的输入 token 数 |
| --prefill-chunk | 64 | 每请求每轮 prefill 上限 |
| --max-tokens | 32 | 新生成 token 上限；0 只校验和分词 |
| --temperature | 0 | 0 为 greedy，正数启用随机采样 |
| --top-k / --top-p | 0 / 1 | 0 表示不限制 top-k；top-p 必须在 (0,1] |
| --seed | 0 | 每个请求自己的随机数状态 |
| --threads | 4 | 后端 CPU 计算线程数 |
| --gpu-layers | 0 | GPU 卸载层数，需要对应构建 backend |

逻辑预算预留完整 prompt 加上 max_tokens - 1 个生成 KV 位置，最后一个采样 token 不再前向。缓存保守按完整 prompt 长度收费，其 token 上限是总 context 的一半；虽然后端共享物理前缀，调度器不利用共享量超售内存。放不下的请求立即拒绝，暂时没有活动槽或预算时排队。FIFO 准入可能出现队首阻塞，这是避免抢占和恢复机制的简化。

--json 的 results 按输入顺序排列，包含文本、token_ids、结束原因，以及 prompt_tokens、completion_tokens、cached_tokens、prefill_tokens。EOS 计入生成数但不作为文本和 token_ids 的可见输出；生成数可能比可见 token_ids 长度多 1。

## HTTP

```bash
curl http://127.0.0.1:1919/health
curl http://127.0.0.1:1919/v1/completions -H "Content-Type: application/json" -d '{"prompt":"Once upon a time","max_tokens":16,"temperature":0}'
curl -N http://127.0.0.1:1919/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":16,"stream":true}'
```

支持字段：model、prompt（文本补全）或 messages（聊天）、max_tokens、temperature、top_p、top_k、seed、stream。messages 只支持 system/user/assistant 的字符串 content。model 可省略；服务只加载启动时指定的一个模型。聊天使用 GGUF 内置模板；无模板或模板不被固定 backend 支持会明确报错，不能随意将消息拼成假模板。

流式结果使用 data: JSON 事件，最后发送 data: [DONE]。字符可能跨 token，服务在 UTF-8 字符完整后输出。断开连接会请求取消；取消在下一次 scheduler step 生效，不能立即打断正在执行的底层 forward。连接、请求队列、输出队列均有上限，过慢客户端会终止而不无限堆积事件。

这是兼容 OpenAI 常见补全结构的有限子集，不宣称完整协议兼容。不支持 stop、n、penalties、logprobs、tools、图片、任意自定义模板；未知字段明确拒绝。默认监听回环地址；没有认证、TLS、持久化和多模型路由。

核心发生模型/采样/KV 错误后实例会失效并尽力清理全部 sequence，后续请求被拒绝，需重启进程。这样避免在部分写入且无法安全回滚的 KV 上继续输出错误答案。

## CPU 构建与发行

默认 MINISGL_CPU_BASELINE=ON 使用通用 CPU 指令集，便于在不同 x86_64 机器验证。若部署 CPU 支持 AVX2/FMA 等，可在 C++ 配置中设 -DMINISGL_CPU_BASELINE=OFF，或在 Rust 构建前设环境变量 MINISGL_CPU_BASELINE=OFF，使用固定 llama.cpp 的优化默认值。请在相同构建选项下做性能比较。GPU 版本还需要对应 CUDA 驱动与运行库。

Rust 当前提供 cargo build 路径；cargo install 不会自动安装旁边的模型桥动态库。发行时复制可执行文件和 adapter，并满足平台/CUDA运行库依赖。
