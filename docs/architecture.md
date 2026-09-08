# Rust 版本：从一次请求理解推理服务

这个分支把调度、压缩 radix 索引、采样和请求生命周期写在 Rust 中，真实模型计算经 `Engine` 交给 llama.cpp。设计围绕三个问题：谁能修改模型状态、哪些 token 已有 KV、资源何时归还。官方机制的对应与差异见 [upstream.md](upstream.md)，这里说明本分支的实际代码。

## 模块的作用、职责和理由

| 模块 | 作用与职责 | 为什么这样设计 |
|---|---|---|
| [types.rs](../src/types.rs) | 请求、配置、统计、`Event`、`Error/Result`；验证采样与容量约束。 | 公共协议只含值类型；HTTP 不需要理解活动 sequence，错误也不会被静默截断。 |
| [engine.rs](../src/engine.rs) | `Engine` trait：分词、token piece、前向、复制和移除 KV sequence。 | 把模型运算与调度策略分开；测试引擎可以验证位置与所有权，真实运行使用 GGUF 后端。 |
| [scheduler.rs](../src/scheduler.rs) | 独占队列、活动请求、sequence ID、逻辑预算和缓存；`step()` 最多执行一次前向，管理取消、完成与错误。 | 一个所有者推进显式状态机，避免把请求状态拆散到后台线程和锁中。 |
| [cache.rs](../src/cache.rs) | 压缩 radix 树、后代代表 sequence、terminal LRU、缓存长度统计。 | 只索引“哪个 sequence 能提供多少前缀 KV”；物理复制与释放仍由 scheduler 决定。 |
| [sampling.rs](../src/sampling.rs) | greedy、temperature、top-k、top-p 与请求级 RNG。 | 策略可独立验证，随机数不会被其他请求的采样消耗。 |
| [llama.rs](../src/llama.rs)、[native/](../native/) | 所有 Rust FFI 集中在 `llama.rs`；桥接模型句柄、tokenizer、logits、聊天模板和物理 KV。 | 把 `unsafe`、整数转换、缓冲区长度和 C 错误处理隔离；核心不依赖 llama.cpp 的内部结构。 |
| [frontend.rs](../src/frontend.rs)、[main.rs](../src/main.rs) | CLI 校验、配置组合、离线运行、JSON 与 UTF-8 展示。 | 可执行程序负责输入输出，库保留可测试的运行时；默认 core 构建不需要模型和 C++ 后端。 |
| [server.rs](../src/server.rs) | 有界 HTTP 连接与命令队列、JSON/SSE、取消通知、响应编码。 | 网络线程处理字节和消息，唯一 model owner 执行聊天模板、分词、调度和模型调用。 |
| [build.rs](../build.rs) | 仅在 `llama` feature 下构建并链接固定版本 C ABI 桥。 | 纯 Rust 核心测试与真实推理构建分开；运行生成必须显式启用真实后端。 |

`Scheduler<E>` 按值拥有引擎，用 `&mut self` 推进状态。`VecDeque<Pending>` 保存 FIFO 队列，`Vec<Active>` 保存稳定的请求顺序。当前 batch 用索引描述计划，先处理所有 logits，再按请求 ID 查找并移除完成项，避免中途删除使后续索引错位。

## 已采样 token 与已计算 KV

正常路径为 `submit → pending → prefill → decode → finished`。`submit()` 做校验和分词后入队；准入才分配 sequence 和活动预算。`max_new_tokens=0` 仍验证 prompt，但 `step()` 直接发 `Length`，不分配 sequence、不执行前向。

| 状态 | 真实字段及含义 |
|---|---|
| pending | `Pending.tokens` 是完整 prompt，`reservation` 是请求最坏 KV 预留；尚无活动 sequence。 |
| prefill | `processed < tokens.len()`；`processed` 是已有 KV 的 token 数，包括命中部分。 |
| decode | `processed >= tokens.len()`；`last_token` 是最新采样但尚未进入 KV 的 token，下一次输入位置正是 `processed`。 |
| finished | 发终结事件、归还预算、移除活动 sequence；正常结束可留下独立 prompt 缓存引用。 |

prompt 长度为 $P$，已采样输出数为 $G$。完成一个调度步后，仍在 decode 的请求满足 $processed=P+G-1$。例如先计算 prompt 的位置 `0..3`，得到第一个输出；下一步才把这个输出放到位置 `4` 并计算其 KV。最终输出已能交付用户，不需要再执行一次前向。

因此输出上限 $N>0$ 的请求预留 $P+N-1$ 个逻辑 KV token；$N=0$ 预留零。EOS 计入 `generated`，但不发文本 token 事件，所以完成事件的生成数可能比可见 `token_ids` 多一个。

`BatchToken` 明确包含 token、position、sequence 和 `logits` 标记。只有最终 prefill 块的最后一个 token、以及 decode 的 token 请求 logits。中间块只计算 KV，不采样，也不消耗 RNG。后端返回行数和顺序必须与这些标记对应；行数不符属于模型协议错误。

## batch 预算和公平性边界

每轮先清理取消与零输出请求，再按 FIFO 准入，然后组装一个混合 batch：

1. 每个 decode 请求分配一个 token 位置。
2. 每个活动 prefill 请求先得到一个 token。
3. 反复按活动顺序遍历 prefill，每次再分一个 token，直到预算用完或达到各自 `prefill_chunk_size` 上限。
4. 执行一次 `forward()`；成功后推进 `processed`，只对需要 logits 的请求采样。

配置约束 `max_active <= max_batch_tokens` 使所有活动请求每轮都能前进一步。decode 不会被新的长 prompt 挤出 batch；额外配额逐个分发，避免第一个长 prefill 一次取走全部余量。每轮仍从活动顺序开头遍历，无法均分的余数偏向前面的请求，没有跨轮旋转起点。

这不是延迟服务等级保证。一次大 prefill 前向仍会让本轮 decode 的结果更晚返回，tokenizer 和 CPU 采样也占用 owner。FIFO 队首预算不足时，后面较小请求不能绕过它。这里不实现抢占和优先级队列；chunk/batch 大小应通过首 token 延迟、每 token 延迟和吞吐实验选择。

## 压缩 radix、部分命中和独立 sequence

`Node.edge` 保存一段 token，`children` 用 `BTreeMap` 按首 token 找分支。分叉时拆边；删除后合并没有 terminal 的单子边。`terminal` 指向完整 prompt 的缓存 sequence，`representative` 指向子树内任意仍存活的缓存 sequence，所以查询停在边中间也能找到实际 KV 来源。

比如缓存 `[10,20,30,40]`，新 prompt `[10,20,99]` 可以命中前两个 token。命中不要求落在完整 prompt 的边界。完全相同的 prompt 也只查前 $P-1$ 个 token，保留最后一个 token 重算以获取本请求 logits。

每个缓存条目保留一个独立 sequence ID，只包含完整 prompt 的 `[0,P)` KV 引用；不保存生成后缀。新请求先确保预算，再查前缀，并复制到自己的空 sequence。后端保证移除来源后目标仍有效，因此缓存可以独立淘汰，不必在 radix 每条边维护活动请求引用计数。

prompt 完整 prefill 后尝试缓存；正常结束时先归还当前逻辑预留，在仍持有活动 sequence 时再尝试一次，然后释放活动 sequence。取消不做这次补存；此前已经缓存的完整 prompt 可以继续存在。缓存超出条目数、token 预算或 ID 容量时按 LRU 淘汰整个缓存 sequence，不抢占活动请求。

LRU 用 `BTreeSet<(stamp, sequence)>`，命中刷新来源条目的时间戳，插入/刷新约 $O(\log E)$，其中 $E$ 为条目数。树查询只比较所走路径的 token；子分支查找带有有序映射成本。压缩树减少索引节点，不等于自动减少物理 KV。`PrefixCache::token_count()` 和 `Stats.cached_tokens` 都累计完整缓存 prompt 长度，共同前缀也重复计费。

## 保守预算不等于显存测量

令 $R$ 为所有活动请求的最坏 KV 预留之和，$C$ 为缓存完整 prompt 长度之和，$B$ 为 `max_kv_tokens`，调度器保持：

$$
R+C\le B.
$$

缓存命中不抵扣活动请求的完整预算。即使 llama.cpp 在物理层共享 KV cell，这里也保守计费。活动请求与缓存共享一个有界 sequence ID 池；每个 ID 都被 `used` 位图追踪，包括复制失败时尚未登记缓存的目标。

物理内存的布局、分配和张量运算由 llama.cpp 管理；桥层启用 unified KV，并核对实际 context、batch、sequence 容量。逻辑 token 数不是显存字节，也不涵盖权重、logits 和临时工作区。应用通过 `--context-size` 配置总共享 token 容量，不能把它理解为每条并发请求都独享这个长度。

## 取消与 fail-closed

`cancel()` 标记请求，下一个 `step()` 在安排新前向前处理。已经执行中的同步前向不被抢占，取消延迟受当前模型调用时长影响。成功释放物理引用后才把 ID 放回 free 池。

`forward`、KV copy/remove、采样和 token piece 错误可能意味着后端已经部分写入。`fail_all()` 因此终结所有未完成请求，并逐个清理 `used` ID，把失败复制目标也纳入清理；错误文本包含清理失败信息。实例被 `poisoned` 标记后拒绝后续生成，没有重置继续使用的入口。

清理失败的 ID 不会重新进入 free 池，`Drop` 会再尝试释放；正常析构同样清理仍被占用的 sequence。Rust 的所有权减少悬空引用，但不能自动回滚外部 C++ 已经修改的模型内存，这就是必须明确错误状态的原因。

## HTTP 并发只传递消息

服务用标准库 TCP 实现有限的 HTTP/1.0、HTTP/1.1 接口，每连接处理一个请求后关闭。8 个 worker、32 个等待 socket、64 个到 model owner 的命令位置，以及每请求 256 个事件位置都有界。header 上限 16 KiB、body 上限 1 MiB，读写有超时。这里没有为每个请求创建推理线程，也没有在模型前向外层加一把全局大锁。

owner 每轮最多接收有限数量命令，然后继续推进调度，持续到来的 HTTP 请求不会阻止已有请求执行。worker 等 owner 完成分词和准入确认，再发送 SSE 响应头，便于对非法输入、满队列和不可用后端返回明确错误。`/health` 读取 owner 发布的健康标志。

worker 等待事件时检测断连，`CancelOnDrop` 用 `AtomicBool` 保证 I/O 提前退出仍通知 owner。发送事件采用 `try_send`；慢客户端填满队列时取消订阅和请求，不让网络写入阻塞模型。已开始的 SSE 通过流内错误报告后结束，不能再更改先前的 HTTP 状态码。

`Utf8Output` 保存跨 token 的不完整 UTF-8 尾部，只输出完整字符，终结时处理残缺字节。token 事件保留 `Vec<u8>`，把字节变成文本的责任留给 frontend；一个 token 不必等于一个字符。

## 采样的成本与 seed 边界

greedy 扫描词表一次，时间 $O(V)$，没有候选数组分配；最大值并列时取较小 token ID。正温度先减最大 logit 再做 double 精度指数，防止溢出。NaN、正无穷和全被屏蔽的 logits 返回错误，负无穷允许作为屏蔽值。

不筛选时，构造候选和随机抽样均为 $O(V)$。top-k 通过 `select_nth_unstable_by` 选出候选，不为它排序整个词表；只有 top-p 小于 1 才把保留候选按概率排序，最坏 $O(V\log V)$，有 top-k 时排序范围缩到 $K$。nucleus 选到累计概率达到阈值的最短前缀。

每个请求持有自己的 SplitMix64 状态。无关请求不会消费其随机数，中间 prefill 块也不采样。C++ 使用不同 RNG 和候选实现，因此相同 seed 不保证两个语言的随机结果相同；硬件、量化和浮点后端变化也可能改变 logits。跨语言验证优先使用相同 GGUF 和后端配置的 greedy token 序列。复制 CPU logits 和原生 CPU 采样是清晰的实现，也可能成为吞吐瓶颈。

## FFI 安全约束

`llama.rs` 用 `NonNull` 持有不透明模型句柄，`Drop` 唯一释放它；没有 `Clone/Copy`。`PhantomData<Cell<()>>` 使引擎不能共享为 `Sync`；明确的 `Send` 实现只允许移动所有权，不能让多个线程同时操作同一 context。KV 修改需要 `&mut self`。

C ABI 只暴露固定宽整数、指针与长度、状态码；Rust 不复制 llama.cpp 私有结构。桥层捕获 C++ 异常，Rust 立即把线程局部错误文本复制为 `Error`。分词、piece、模板都先查询长度再写入；位置转成 int32 和 logits 缓冲区乘法都检查范围。`unsafe` 注释说明指针的存活期和容量依据。

这层桥适配固定版本 llama.cpp；不承诺任意未来版本兼容。测试后端实现同一个 trait，但不属于生产生成路径。未启用 `llama` feature 时可以测试核心，不能用伪模型替代真实推理。

## 阅读顺序和可操作实验

先读 `types.rs → engine.rs → scheduler.rs`，手算一个 4 token prompt 的两次前向；再读 `cache.rs → sampling.rs → llama.rs/native → frontend.rs/server.rs`。[tests/core.rs](../tests/core.rs) 展示测试引擎如何检查 batch 位置、真实 KV 引用生命周期和失败注入。

| 实验 | 操作 | 应观察什么 |
|---|---|---|
| 分块与批处理 | 固定 GGUF，`--temperature 0 --json`；比较单请求与多个 `--prompt`，把 `--prefill-chunk` 改为 1、7、64。 | 正常回归保持 greedy token 序列；前向次数和延迟改变。 |
| 完整与部分前缀 | 同一进程用 `--max-sequences 1` 顺序输入相同 prompt，再输入只有前半相同的 prompt。 | 完整命中 `cached_tokens=P-1`、`prefill_tokens=1`；部分命中减少实际前向 token，不能只看树的命中计数。 |
| 淘汰 | `--cache-sequences 1` 输入 A、B、A，与禁用缓存比较。 | 被淘汰部分需要重算；复用 ID 后输出不被另一序列污染。 |
| 公平性与预算 | 核心测试启动 decode，再加入多个长 prefill；缩小 batch 和总 KV。 | 活动请求每轮进展，余量逐个分发；准入仍有 FIFO 队首等待。 |
| 取消与失败 | 中断流式客户端；用测试引擎令 forward/copy/remove 返回错误。 | 取消回收资源；后端异常使实例失效，失败清理的 ID 不被复用。 |
| 采样 | 对固定 logits 检查 greedy、top-k、top-p，加入无关并发请求。 | 请求 RNG 不互相干扰；不要求跨语言随机结果完全相同。 |

[tests/conformance.py](../tests/conformance.py) 使用真实 GGUF 比较单请求/批处理、chunk、完整/部分前缀、LRU 淘汰和零输出，也能同时验证两个语言的二进制。核心测试证明状态机行为，模型集成测试证明真正接通计算后端，两者都不能互相替代。
