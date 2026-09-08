#pragma once

// 作用：把模型 logits 转成下一个 token，集中处理 greedy、temperature、top-k、top-p。
// 职责：校验采样参数、过滤不可选分数并执行一次选择，不管理请求、模型或 KV 缓存。
// 设计原因：随机数发生器由请求持有并显式传入，批次重排不会改变其他请求的随机序列，
// 也无需全局锁；采样逻辑因此可以独立验证和替换。

#include "minisgl/types.hpp"

#include <random>
#include <span>

namespace minisgl {

void validate_sampling(const SamplingParams &params);

// NaN 或 +inf 表示损坏的模型输出，抛出 std::invalid_argument；-inf 作为 mask 不可选。
// temperature == 0 时返回分数最大的 token，平局选最小 id，且不消耗 rng。
// top_k == 0 表示不过滤；top-p 在 top-k 后的归一化分布上保留最小概率前缀。
// 空 logits 或没有可选 token 时抛出 std::invalid_argument。
// logits 只在调用期间借用；既可以传 vector，也可以传数组或其中一段。
Token sample_token(std::span<const float> logits, const SamplingParams &params,
                   std::mt19937_64 &rng);

} // namespace minisgl
