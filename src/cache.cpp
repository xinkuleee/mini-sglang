#include "minisgl/cache.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace minisgl {

// 每个非 root 节点保存一条压缩边，避免逐 token 分配节点。representative
// 指向子树内仍存活的 terminal，使边中途命中也能 O(1) 找到 KV 提供者。
struct PrefixCache::Node {
    std::vector<Token> edge;
    Node *parent = nullptr;
    std::unordered_map<Token, std::unique_ptr<Node>> children;
    std::optional<SequenceId> sequence;
    std::list<SequenceId>::iterator lru;
    Node *representative = nullptr;

    Node(std::vector<Token> edge_tokens, Node *parent_node)
        : edge(std::move(edge_tokens)), parent(parent_node) {}
};

PrefixCache::PrefixCache() : root_(std::make_unique<Node>(std::vector<Token>{}, nullptr)) {}
PrefixCache::~PrefixCache() = default;

std::size_t PrefixCache::size() const noexcept { return terminals_.size(); }
std::size_t PrefixCache::node_count() const noexcept { return nodes_; }
std::size_t PrefixCache::token_count() const noexcept { return tokens_; }

void PrefixCache::clear() noexcept {
    root_->children.clear();
    root_->representative = nullptr;
    terminals_.clear();
    lru_.clear();
    nodes_ = 1;
    tokens_ = 0;
}

void PrefixCache::touch(Node &node) { lru_.splice(lru_.end(), lru_, node.lru); }

void PrefixCache::refresh(Node &node) {
    node.representative =
        node.sequence
            ? &node
            : (node.children.empty() ? nullptr : node.children.begin()->second->representative);
}

void PrefixCache::refresh_ancestors(Node *node) {
    for (; node != nullptr; node = node->parent) {
        refresh(*node);
    }
}

PrefixCache::Node *PrefixCache::find_exact(std::span<const Token> tokens) const {
    Node *node = root_.get();
    std::size_t offset = 0;
    while (offset < tokens.size()) {
        const auto child_it = node->children.find(tokens[offset]);
        if (child_it == node->children.end()) {
            return nullptr;
        }
        Node *child = child_it->second.get();
        if (child->edge.size() > tokens.size() - offset ||
            !std::equal(child->edge.begin(), child->edge.end(), tokens.begin() + offset)) {
            return nullptr;
        }
        offset += child->edge.size();
        node = child;
    }
    return node;
}

std::optional<PrefixMatch> PrefixCache::find(std::span<const Token> tokens,
                                             std::size_t max_tokens) {
    const auto limit = std::min(tokens.size(), max_tokens);
    Node *node = root_.get();
    Node *source = nullptr;
    std::size_t matched = 0;
    while (matched < limit) {
        const auto child_it = node->children.find(tokens[matched]);
        if (child_it == node->children.end()) {
            break;
        }
        Node *child = child_it->second.get();
        const auto available = std::min(child->edge.size(), limit - matched);
        std::size_t common = 0;
        while (common < available && child->edge[common] == tokens[matched + common]) {
            ++common;
        }
        matched += common;
        source = child->representative;
        if (common != child->edge.size()) {
            break;
        }
        node = child;
    }
    if (matched == 0 || source == nullptr) {
        return std::nullopt;
    }
    touch(*source);
    return PrefixMatch{*source->sequence, matched};
}

std::optional<SequenceId> PrefixCache::insert(std::span<const Token> tokens, SequenceId sequence) {
    if (tokens.empty() || sequence < 0) {
        throw std::invalid_argument(
            "prefix cache requires a nonempty prompt and nonnegative sequence");
    }
    const auto existing_sequence = terminals_.find(sequence);
    if (existing_sequence != terminals_.end()) {
        if (find_exact(tokens) != existing_sequence->second) {
            throw std::invalid_argument("sequence already belongs to another cached prompt");
        }
        touch(*existing_sequence->second);
        return std::nullopt;
    }

    Node *node = root_.get();
    std::size_t offset = 0;
    while (offset < tokens.size()) {
        const auto child_it = node->children.find(tokens[offset]);
        if (child_it == node->children.end()) {
            auto leaf = std::make_unique<Node>(
                std::vector<Token>(tokens.begin() + offset, tokens.end()), node);
            Node *leaf_ptr = leaf.get();
            node->children.emplace(leaf->edge.front(), std::move(leaf));
            ++nodes_;
            tokens_ += tokens.size() - offset;
            node = leaf_ptr;
            break;
        }

        Node *child = child_it->second.get();
        const auto available = std::min(child->edge.size(), tokens.size() - offset);
        std::size_t common = 0;
        while (common < available && child->edge[common] == tokens[offset + common]) {
            ++common;
        }
        if (common < child->edge.size()) {
            auto branch = std::make_unique<Node>(
                std::vector<Token>(child->edge.begin(), child->edge.begin() + common), node);
            const Token suffix_key = child->edge[common];
            // 先分配 child 槽位，后转移所有权；分裂不复制整个历史 prompt。
            branch->children.emplace(suffix_key, nullptr);
            child->edge.erase(child->edge.begin(), child->edge.begin() + common);
            child->parent = branch.get();
            branch->representative = child->representative;
            branch->children.at(suffix_key) = std::move(child_it->second);
            child_it->second = std::move(branch);
            child = child_it->second.get();
            ++nodes_;
        }
        offset += common;
        node = child;
    }

    const auto replaced = node->sequence;
    if (replaced) {
        terminals_.emplace(sequence, node);
        terminals_.erase(*replaced);
        *node->lru = sequence;
        node->sequence = sequence;
        touch(*node);
    } else {
        lru_.push_back(sequence);
        try {
            terminals_.emplace(sequence, node);
        } catch (...) {
            lru_.pop_back();
            compact_ancestors(node);
            throw;
        }
        node->lru = std::prev(lru_.end());
        node->sequence = sequence;
    }
    refresh_ancestors(node);
    return replaced;
}

// 删除 terminal 后立即清理空枝、合并非 terminal 的单分支。其余 terminal
// 的地址和 sequence 保持稳定，因此共享前缀的删除不会使其他 KV 句柄失效。
void PrefixCache::compact_ancestors(Node *node) {
    while (node != root_.get()) {
        Node *parent = node->parent;
        const Token key = node->edge.front();
        if (!node->sequence && node->children.empty()) {
            tokens_ -= node->edge.size();
            --nodes_;
            parent->children.erase(key);
            node = parent;
            continue;
        }
        if (!node->sequence && node->children.size() == 1) {
            Node *child_ptr = node->children.begin()->second.get();
            std::vector<Token> merged;
            merged.reserve(node->edge.size() + child_ptr->edge.size());
            merged.insert(merged.end(), node->edge.begin(), node->edge.end());
            merged.insert(merged.end(), child_ptr->edge.begin(), child_ptr->edge.end());
            auto child = std::move(node->children.begin()->second);
            child->edge = std::move(merged);
            child->parent = parent;
            parent->children.at(key) = std::move(child);
            --nodes_;
            node = child_ptr;
        }
        refresh(*node);
        node = parent;
    }
    refresh(*root_);
}

bool PrefixCache::erase_sequence(SequenceId sequence) {
    const auto terminal_it = terminals_.find(sequence);
    if (terminal_it == terminals_.end()) {
        return false;
    }
    Node *node = terminal_it->second;
    lru_.erase(node->lru);
    terminals_.erase(terminal_it);
    node->sequence.reset();
    compact_ancestors(node);
    return true;
}

std::optional<SequenceId> PrefixCache::evict_lru() {
    if (lru_.empty()) {
        return std::nullopt;
    }
    const auto victim = lru_.front();
    erase_sequence(victim);
    return victim;
}

void PrefixCache::assert_invariants() const {
    const auto require = [](bool valid, const char *message) {
        if (!valid) {
            throw std::logic_error(std::string("prefix cache invariant: ") + message);
        }
    };
    require(root_ != nullptr, "missing root");
    require(root_->parent == nullptr && root_->edge.empty() && !root_->sequence,
            "root must be empty and nonterminal");
    require(lru_.size() == terminals_.size(), "LRU and terminal sizes differ");

    std::unordered_set<SequenceId> lru_sequences;
    for (auto it = lru_.begin(); it != lru_.end(); ++it) {
        require(lru_sequences.insert(*it).second, "duplicate sequence in LRU");
        const auto terminal_it = terminals_.find(*it);
        require(terminal_it != terminals_.end(), "LRU refers to absent terminal");
        require(terminal_it->second->lru == it, "terminal has incorrect LRU iterator");
    }

    std::vector<const Node *> pending{root_.get()};
    std::size_t actual_nodes = 0;
    std::size_t actual_tokens = 0;
    std::size_t actual_terminals = 0;
    while (!pending.empty()) {
        const Node *node = pending.back();
        pending.pop_back();
        ++actual_nodes;
        actual_tokens += node->edge.size();
        if (node != root_.get()) {
            require(!node->edge.empty() && node->parent != nullptr, "invalid compressed edge");
            require(node->sequence || node->children.size() >= 2, "uncompressed or empty branch");
        }
        if (node->sequence) {
            ++actual_terminals;
            const auto terminal_it = terminals_.find(*node->sequence);
            require(terminal_it != terminals_.end() && terminal_it->second == node,
                    "terminal missing from sequence index");
            require(node->representative == node, "terminal must represent itself");
        } else if (node->children.empty()) {
            require(node->representative == nullptr, "empty subtree has representative");
        } else {
            bool represented_by_child = false;
            for (const auto &child : node->children) {
                represented_by_child |=
                    child.second && child.second->representative == node->representative;
            }
            require(node->representative != nullptr && represented_by_child,
                    "branch representative is outside subtree");
        }
        for (const auto &child : node->children) {
            require(child.second && !child.second->edge.empty(), "empty child");
            require(child.first == child.second->edge.front(), "incorrect child key");
            require(child.second->parent == node, "incorrect parent pointer");
            pending.push_back(child.second.get());
        }
    }
    require(actual_nodes == nodes_, "incorrect node count");
    require(actual_tokens == tokens_, "incorrect token count");
    require(actual_terminals == terminals_.size(), "incorrect terminal count");
}

} // namespace minisgl
