// 作用与职责见 sampling.hpp。本实现先缩小候选集合，再用 double 精度的稳定 softmax
// 计算权重；没有 top-k/top-p 筛选时保持线性复杂度，避免每一步都排序整个词表。

#include "minisgl/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace minisgl {
namespace {

struct Candidate {
    Token token;
    float logit;
    double weight = 0.0;
};

bool precedes(const Candidate &lhs, const Candidate &rhs) {
    return lhs.logit > rhs.logit || (lhs.logit == rhs.logit && lhs.token < rhs.token);
}

bool selectable(float logit) { return std::isfinite(logit); }

} // namespace

void validate_sampling(const SamplingParams &params) {
    if (!std::isfinite(params.temperature) || params.temperature < 0.0f) {
        throw std::invalid_argument("sampling temperature must be finite and nonnegative");
    }
    if (!std::isfinite(params.top_p) || params.top_p <= 0.0f || params.top_p > 1.0f) {
        throw std::invalid_argument("sampling top_p must be finite and in (0, 1]");
    }
}

Token sample_token(std::span<const float> logits, const SamplingParams &params,
                   std::mt19937_64 &rng) {
    validate_sampling(params);
    if (logits.empty()) {
        throw std::invalid_argument("cannot sample an empty logits vector");
    }
    if (logits.size() - 1 > static_cast<std::size_t>(std::numeric_limits<Token>::max())) {
        throw std::length_error("logits vocabulary exceeds the token id range");
    }

    bool found = false;
    Token best_token = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (std::isnan(logits[i]) || logits[i] == std::numeric_limits<float>::infinity()) {
            throw std::invalid_argument("logits contain NaN or positive infinity");
        }
        if (selectable(logits[i]) && (!found || logits[i] > best_logit)) {
            found = true;
            best_token = static_cast<Token>(i);
            best_logit = logits[i];
        }
    }
    if (!found) {
        throw std::invalid_argument("logits contain no selectable token");
    }
    if (params.temperature == 0.0f) {
        return best_token;
    }

    std::vector<Candidate> candidates;
    candidates.reserve(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (selectable(logits[i])) {
            candidates.push_back({static_cast<Token>(i), logits[i], 0.0});
        }
    }

    if (params.top_k != 0 && params.top_k < candidates.size()) {
        std::partial_sort(candidates.begin(), candidates.begin() + params.top_k, candidates.end(),
                          precedes);
        candidates.resize(params.top_k);
    } else if (params.top_p < 1.0f) {
        std::sort(candidates.begin(), candidates.end(), precedes);
    }

    double total_weight = 0.0;
    for (auto &candidate : candidates) {
        // 先相减再除 temperature，避免极小温度导致正的指数溢出。
        candidate.weight =
            std::exp((static_cast<double>(candidate.logit) - static_cast<double>(best_logit)) /
                     static_cast<double>(params.temperature));
        total_weight += candidate.weight;
    }

    if (params.top_p < 1.0f) {
        const double threshold = static_cast<double>(params.top_p) * total_weight;
        double kept_weight = 0.0;
        std::size_t kept = 0;
        do {
            kept_weight += candidates[kept++].weight;
        } while (kept < candidates.size() && kept_weight < threshold);
        candidates.resize(kept);
        total_weight = kept_weight;
    }
    if (candidates.size() == 1) {
        return candidates.front().token;
    }

    // 显式使用 mt19937_64 的高 53 位得到 [0, 1)，避免不同标准库的
    // uniform_real_distribution/discrete_distribution 算法差异。
    const double unit = static_cast<double>(rng() >> 11) * 0x1.0p-53;
    const double target = unit * total_weight;
    double cumulative = 0.0;
    Token last_positive = best_token;
    for (const auto &candidate : candidates) {
        if (candidate.weight > 0.0) {
            last_positive = candidate.token;
            cumulative += candidate.weight;
            if (target < cumulative) {
                return candidate.token;
            }
        }
    }
    // 浮点乘法/求和的舍入可能让 target 恰好落在末端，绝不回退到零权重项。
    return last_positive;
}

} // namespace minisgl
