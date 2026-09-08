#pragma once

#include "minisgl/types.hpp"

#include <cstddef>
#include <list>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>

namespace minisgl {

// 前缀缓存只管理「完整 prompt -> 后端 sequence」的索引，不拥有 KV 张量。
// 压缩 radix 边共享 token 前缀；terminal 保存可借用的独立 sequence。
// scheduler 决定复制多少 KV、何时释放 sequence，避免缓存与后端互相依赖。
// 本类由单线程 scheduler 独占；terminal LRU 决定驱逐顺序。
struct PrefixMatch {
    SequenceId sequence;
    std::size_t matched_tokens; // 可以复制的 [0, matched_tokens) KV 范围。
};

class PrefixCache {
  public:
    PrefixCache();
    ~PrefixCache();
    PrefixCache(const PrefixCache &) = delete;
    PrefixCache &operator=(const PrefixCache &) = delete;
    PrefixCache(PrefixCache &&) = delete;
    PrefixCache &operator=(PrefixCache &&) = delete;

    // 返回最长公共前缀；命中会刷新提供 KV 的完整 prompt 的 LRU。
    // caller 通常传 prompt.size() - 1，保留最后一个 token 来重算 logits。
    std::optional<PrefixMatch> find(std::span<const Token> tokens, std::size_t max_tokens);

    // 只插入 KV 已经完整计算的 prompt。相同 prompt 替换时返回旧 sequence，
    // 由 caller 释放；同一 sequence 不允许对应不同 prompt。
    // 空 prompt、负 sequence 或 sequence 被不同 prompt 占用时抛 invalid_argument。
    // 输入只在调用期间借用；树持有所需 token 的副本，不保留 span。
    std::optional<SequenceId> insert(std::span<const Token> tokens, SequenceId sequence);
    std::optional<SequenceId> evict_lru();
    bool erase_sequence(SequenceId sequence);
    // 仅清空索引，不回收后端 KV；失败清理时 caller 已逐一释放 sequence。
    // 无重新分配、无边合并，便于 bad_alloc 之后可靠丢弃缓存状态。
    void clear() noexcept;

    std::size_t size() const noexcept;
    std::size_t node_count() const noexcept;  // 包含不存 token 的 root。
    std::size_t token_count() const noexcept; // 所有压缩边的 token 总数。

    // 调试/测试接口：检查树、terminal 索引、LRU 和统计量，不受 NDEBUG 影响。
    // 发现内部不一致时抛 logic_error。
    void assert_invariants() const;

  private:
    struct Node;
    std::unique_ptr<Node> root_;
    std::unordered_map<SequenceId, Node *> terminals_;
    std::list<SequenceId> lru_; // 最旧 -> 最新；splice 刷新为 O(1)。
    std::size_t nodes_ = 1;
    std::size_t tokens_ = 0;

    Node *find_exact(std::span<const Token> tokens) const;
    void touch(Node &node);
    void refresh(Node &node);
    void refresh_ancestors(Node *node);
    void compact_ancestors(Node *node);
};

} // namespace minisgl
