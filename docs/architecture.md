# C++ 版本：从一次请求理解推理服务

这个分支把调度、前缀索引、采样和请求生命周期写在 C++ 中，真实模型计算经 `Engine` 交给 llama.cpp。阅读重点是：谁决定执行、哪些 token 已有 KV、资源何时归还。与官方实现的对应和范围见 [upstream.md](upstream.md)，构建与已完成的验证见项目 README 和验证文档。

## 模块为什么这样拆

| 模块 | 作用与职责 | 为什么这样设计 |
|---|---|---|
| [types.hpp](../include/minisgl/types.hpp)、[types.cpp](../src/types.cpp) | 定义请求、采样参数、事件、配置和统计；验证容量关系。 | 值类型是模块协议，HTTP 和模型层都无需认识内部请求对象。 |
| [scheduler.hpp](../include/minisgl/scheduler.hpp)、[scheduler.cpp](../src/scheduler.cpp) | 独占 pending/active 队列、sequence ID、逻辑 KV 预算；每次 `step()` 最多执行一个 batch，处理取消和终结。 | 把资源决策放在一个状态机中，便于检查不变量，避免 HTTP 线程同时修改 KV。`Impl` 隐藏内部容器，不扩散实现依赖。 |
| [cache.hpp](../include/minisgl/cache.hpp)、[cache.cpp](../src/cache.cpp) | 压缩 radix 树索引 token 前缀，维护可用 sequence 的代表节点和 LRU。 | 索引只回答“从哪个 sequence 复用多少 token”；复制和释放物理 KV 由调度器负责，树无需依赖 llama.cpp。 |
| [sampling.hpp](../include/minisgl/sampling.hpp)、[sampling.cpp](../src/sampling.cpp) | 从 logits 执行 greedy、temperature、top-k、top-p，使用请求自己的 RNG。 | 采样策略可以单独验证，其他请求不会消费本请求的随机数。 |
| [engine.hpp](../include/minisgl/engine.hpp) | 定义 tokenize、piece、forward、KV sequence copy/remove 的窄接口。 | 测试可以检查调度产生的真实位置和顺序；生产后端仍必须计算模型 logits。 |
| [llama_engine.cpp](../src/llama_engine.cpp)、[native/](../native/) | RAII 持有模型句柄，把 C++ 容器转换成 C ABI；桥接 GGUF、tokenizer、batch 前向及 KV 引用。 | 把长度检查、C 字符串和错误转换收在边界；调度器无需感知 llama.cpp 的结构布局与版本变化。 |
| [main.cpp](../src/main.cpp) | 解析 CLI、组合后端与调度器、运行离线请求、输出结果。 | 组合根统一配置模型实际容量与调度逻辑容量；核心不依赖 JSON 或终端。 |
| [server.cpp](../src/server.cpp)、[server.hpp](../include/minisgl/server.hpp) | HTTP/JSON/SSE、聊天模板请求、UTF-8 拼接、有界通道、断连通知。 | 网络并发只搬运消息；模型 owner 执行调度、分词和聊天模板处理。 |

`Scheduler` 借用 `Engine&`，所以后端必须活得更久；CLI 的构造顺序满足这一点。调度器内部用 `list<unique_ptr<Active>>` 保持活动请求地址稳定，batch 计划中的指针不会因另一个请求完成而失效。调度器和 `LlamaEngine` 禁止复制，防止重复资源所有权。

## C++20 的使用边界

主运行时通过 CMake 的 PUBLIC cxx_std_20 要求向调用者传播标准版本；native/ 的 C ABI 桥和 llama.cpp 保持 C++17。C ABI 不暴露 STL 类型，两层无须采用同一标准。

forward 的 batch、采样的 logits、缓存输入与聊天消息使用 std::span<const T>，表示仅在同步调用期间借用连续数据。可传 vector、array 或其中一段，不要求调用者创建临时 vector；调用者必须保证底层数据在调用结束前存活且不被改动。后端不得保存 batch 视图，缓存插入仍复制所需 token，返回的 logits 仍由 vector 拥有。原来的 const vector& 已不复制数据，因此不把接口改动宣称为自动加速。

服务拥有者和退出监控使用 std::jthread/std::stop_token；线程对象通过 RAII 请求停止并等待退出，显式 stop 保证模型拥有者在依赖销毁之前完成清理。停止是协作式的：必须先等同步模型前向返回，不能将 stop_token 当作 GPU 中断。

## 请求状态：采样完成不等于 KV 已计算

正常路径是 `submit → pending → prefill → decode → finished`。`submit()` 完成校验和分词后入队；只有准入时才占活动 sequence 和 KV 预算。`max_new_tokens=0` 仍校验、分词，但下一步直接返回 `length`，不分配 sequence、不执行模型。

| 状态 | 关键字段与下一步 |
|---|---|
| pending | 保存完整 prompt 和最坏预算；尚无活动 sequence。 |
| prefill | `prefilled < prompt.size()`；输入尚未计算的 prompt 片段。`cached + computed_prefill == prefilled`。 |
| decode | prompt 已计算且 `generated > 0`；把 `last_token` 放到位置 `prompt.size() + generated - 1`，计算后再采样一个 token。 |
| finished | 发送终结事件，归还活动预算和 sequence；正常结束时可以留下独立的 prompt 缓存引用。 |

假设 prompt 有 $P$ 个 token，已采样 $G$ 个输出。完成一个调度步后，仍在 decode 的请求已有 $P+G-1$ 个位置的 KV，最新采样 token 尚未前向。比如 $P=4$：计算位置 `0..3` 后采样第一个输出；下一步才把该输出放到位置 `4`。达到输出上限时，最后一个输出不再计算 KV。

因此生成上限为 $N>0$ 时，准入预留 $P+N-1$ 个逻辑 KV token；$N=0$ 的预留为零。EOS 本身计入已采样数量，但默认不生成文本事件；完成事件的 `completion_tokens` 可能比可见 `token_ids` 多一个。

每个 `BatchToken` 显式包含 token、position、sequence、是否需要 logits。只有最终 prefill 块的最后一个 token 和每个 decode token 请求 logits。中间 prefill 块只写 KV，不采样，也不消耗随机数。`Engine::forward()` 只返回被请求的 logits 行，顺序必须与输入一致；行数不符会使实例停止服务。

## 一个 step 怎样分配计算预算

每轮先处理取消和零输出请求，再按 FIFO 尝试准入，然后生成一个混合 batch：

1. 每个已有 decode 请求占一个 token 位置。
2. 每个活动 prefill 请求先保底一个 token。
3. 剩余预算按轮转起点分给 prefill，单请求最多 `prefill_chunk_size` 个 token。
4. 执行一次前向，成功后推进逻辑位置，对需要 logits 的请求采样并发出事件。

配置要求 `max_running_requests <= max_batch_tokens`，因此前两步总能容纳所有活动请求。batch 中先放 decode，再按活动顺序放 prefill；轮转改变的是额外配额，而不是请求的语义位置。长 prompt 不会占走现有 decode 的位置，活动 prefill 也不会因另一个长 prompt 而完全停步。

这只保证每轮有进展，不保证相同延迟。大 prefill batch 仍会增加一次模型调用的时间；CPU 分词、聊天模板、采样也在 owner 上执行。FIFO 准入在队首没有预算时停止，后方较小请求可能等待。这里没有抢占、优先级或跨租户延迟保证。调整 chunk/batch 预算是在首 token 等待、每 token 延迟和吞吐之间做实验。

## Radix 索引怎样对应真实 KV

树的一条边保存一段 token，公共前缀共享索引节点。插入分叉时拆边；删除后合并没有 terminal 的单子节点。terminal 对应一个完整 prompt 的独立 sequence ID。每个节点维护仍然存活的后代 `representative`，所以命中边中途也能找到可复制 KV 的来源。

例如缓存 token 序列 `[10,20,30,40]`，新 prompt 是 `[10,20,99]`，索引返回前两个 token 以及缓存 sequence。它不要求这两个 token 恰好是 terminal，也不需要重算公共部分。完全相同 prompt 的命中仍被限制到 $P-1$，必须重算最后一个 token，才能得到当前请求的下一 token logits。

缓存建立和使用遵循以下顺序：

1. prompt 完整 prefill 后，若预算和 ID 允许，申请缓存 sequence，把 `[0,P)` 的 KV 引用复制进去，再登记索引。
2. 新请求先腾出准入空间，再查询前缀、复制到自己的空 sequence。复制后，即使缓存来源被淘汰，活动请求的引用仍须有效。
3. 正常结束时先归还当前请求的逻辑预留，在仍持有源 sequence 时再尝试缓存，随后释放活动 sequence。取消不做这次补存；此前已建立的完整 prompt 缓存可以保留。
4. 缓存有条目、token 和 ID 限制，压力下按 terminal LRU 淘汰整个缓存 sequence。命中刷新提供 KV 的 terminal；不会淘汰活动 sequence。

`copy_sequence` 复制的是独立 sequence 的 KV 引用，不承诺复制一整份 tensor。llama.cpp 可以让多个 sequence 指向同一 KV cell。这也解释了为什么这里不需要给 radix 的每条边维护活动请求引用计数：活动请求已经取得自己的后端引用。

两种统计不可混淆：`PrefixCache::token_count()` 是压缩边保存的索引 token 总数；`SchedulerStats::cache_tokens` 是各缓存完整 prompt 长度之和。调度预算使用后者，共同前缀也重复计费。

## 逻辑预算与物理内存的边界

令 $R$ 为活动请求的最坏 KV 预留之和，$C$ 为完整缓存 prompt 长度之和，$B$ 为 `max_total_tokens`。调度器持续约束：

$$
R+C\le B.
$$

缓存命中不会抵扣活动请求的完整预留；物理 KV 即使共享，也按更保守的独立长度准入。健康的稳定状态下，每个 sequence ID 恰好属于 free、active 或 cache；复制尚未完成的目标临时标为 `Transient`，异常清理也会覆盖它。清理失败的 ID 标为 `Quarantined`，不能重新分配。`assert_invariants()` 检查树、所有权、请求 ID 和预算，而不是依赖仅在 debug 生效的断言。

物理 KV 的布局、分配与计算由 llama.cpp 管理；桥层启用 unified KV，并校验实际 context、batch、sequence 容量不少于请求配置。逻辑 token 计数不是显存字节测量，也不包含模型权重、logits 和工作区。若后端因内存或其他原因失败，不能因逻辑预算尚有余量而继续使用未知状态。

## 取消、错误和有界 HTTP

`cancel()` 只标记请求，下一次 `step()` 在安排新计算前终结它。正在执行的同步前向不会被强行中断，取消延迟至少受当前模型调用时长影响。归还 ID 前必须先成功移除该 sequence 的物理引用。

前向、复制、释放、采样或输出 piece 发生异常时，调度器无法确定后端有没有写入部分 KV，因此采用 fail-closed：标记 `healthy=false`，先尽力清理所有占用 sequence，再给未完成请求发错误并清空请求状态，拒绝新请求。清理失败的 sequence 保持 `Quarantined`，通过 `quarantined_sequences` 统计，析构时重试；不能假报为 free。若进程连错误事件都无法分配内存，只能保证清理尝试，不能保证每条错误送达。这里没有“清个错误继续跑”的恢复入口；需要重新创建后端和调度器。

HTTP 使用 cpp-httplib 的 8 个 worker 和 32 个等待连接位置；到 owner 的入站队列上限为 64，每个请求的事件队列上限为 256，调度器 pending 队列另有配置限制。互斥锁仅保护消息队列和统计快照，不覆盖模型前向。先得到 owner 的准入确认，再发送流响应头，可以对无效输入或满队列返回明确 HTTP 错误。

断连、不可写 socket、流结束回调会设置取消标志；owner 每轮读取标志。慢客户端的事件队列满时改为错误并取消请求，避免阻塞所有请求的模型线程。已经发送 SSE 响应头后，后续错误通过流内错误事件报告，不能再更改 HTTP 状态。`/health` 读取 owner 的统计和健康快照。

token piece 是原始字节，可能只含中文字符的一部分；`consume_utf8()` 缓存不完整尾部，只交付完整字符，终结时处理残缺字节。流式和非流式都在传输边界做这件事，核心不把 token 等同于字符。

## 采样成本与可复现边界

词表大小为 $V$。零温度 greedy 扫描一次，时间 $O(V)$，不分配候选数组，相同最大值取较小 token ID。正温度使用 double 权重并先减最大 logit，避免指数溢出；NaN、正无穷和没有有限候选的 logits 被拒绝，负无穷可以屏蔽 token。

不筛选时，候选构造和抽样都是 $O(V)$。top-k 使用 `partial_sort`，约 $O(V\log K)$，保留有序前 $K$ 项；只有 top-p 且未用 top-k 时，需要对整个候选集合排序，最坏 $O(V\log V)$。nucleus 从保留集合中取达到累计概率阈值的最短前缀。

每个请求持有 `mt19937_64`，并从其高 53 位构造均匀随机数。请求间不会互相消耗 RNG，但 C++ 和 Rust 使用不同 RNG 与候选实现，不保证相同 seed 的随机输出跨语言一致。量化、硬件和数值内核变化也可能影响 logits；跨语言回归优先比较同模型、同后端配置下的 greedy token 序列。CPU logits 复制和采样是清晰的教学实现，也是可能的性能瓶颈。

## FFI 为什么保留一层桥

`native/minisgl_bridge.h` 暴露自己的简单 C ABI：不透明句柄、固定宽整数、指针与长度、状态码。C++ 桥捕获异常，保存线程局部错误文本；`LlamaEngine` 立即复制错误并转换为异常。tokenizer、piece 和聊天模板采用先查询长度再写入的协议，位置和 logits 缓冲区大小都有检查。

模型/上下文由桥层 RAII 释放，`LlamaEngine` 再通过析构释放桥句柄。调度器只依赖 `Engine`，可以用测试后端记录每次位置与 KV 操作；替换后端不应改动准入或采样规则。这是固定版本 llama.cpp 的适配边界，不是对任意未来版本 ABI 稳定性的承诺。

## 建议阅读顺序与实验

先读 `types.hpp → engine.hpp → scheduler.cpp`，用一个 4 token prompt 手算两次前向；再读 `cache.cpp → sampling.cpp → llama_engine.cpp/native → main.cpp/server.cpp`。读 [tests/core_tests.cpp](../tests/core_tests.cpp) 时关注测试后端如何检查位置、KV 复制和故障，而不只看输出字符串。

| 实验 | 操作 | 应观察什么 |
|---|---|---|
| 分块与批处理 | 同一 GGUF 用 `--temperature 0 --json`，比较单请求和多个 `--prompt`；把 `--prefill-chunk` 改为 1、7、64。 | 正常回归应保持 greedy token 序列；前向次数变化。不能仅比较自然语言看起来是否合理。 |
| 真正的前缀命中 | 同一进程设置 `--max-sequences 1`，连续提交同一 prompt，再提交共享部分前缀的 prompt。 | 第二次 `cached_tokens=P-1`、`prefill_tokens=1`；部分命中降低实际送入前向的 token 数，输出仍正确。 |
| 缓存淘汰 | `--cache-sequences 1`，顺序运行 A、B、A，并与禁用缓存比较。 | 被淘汰部分需要重算，不能因 ID 复用污染输出；`cache_tokens` 是保守计费。 |
| 公平性 | 核心测试中先启动 decode，再加入多个长 prompt，缩小 batch。 | 每个活动请求每轮至少前进一步；额外 prefill 配额轮转，但单步耗时不会相等。 |
| 取消和故障 | 在流式 HTTP 中断开客户端；核心测试令 forward/copy/remove 返回错误。 | 取消后活动预算归还；后端错误后实例不再接受生成，不能静默产生后续文本。 |
| 采样 | 固定一组人工 logits，分别启用 greedy、top-k、top-p；让无关请求同时运行。 | 采样边界和请求 RNG 隔离正确；不以跨语言随机 token 完全一致作为验收。 |

[tests/conformance.py](../tests/conformance.py) 可对真实 GGUF 自动比较批次、分块、完整/部分前缀、淘汰和零输出，也能同时接收两个语言的可执行文件。核心测试验证状态机，真实模型测试验证计算连接；两者回答的问题不同。
