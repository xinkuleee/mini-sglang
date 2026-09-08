//! 压缩 radix 索引把公共 token 前缀放进一条边；分叉时拆边，删除后合并单子边。
//! 条目拥有完整 prompt 的独立物理序列；树只索引 token，不持有模型内存。每个节点
//! 记录一个仍存活的后代序列，边中间的命中也可直接复制 KV。LRU 与树分离。
use crate::types::{Error, Result};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixMatch {
    pub sequence: u32,
    pub tokens: usize,
}
#[derive(Debug)]
pub struct CacheEntry {
    pub sequence: u32,
    pub tokens: Vec<i32>,
    stamp: u128,
}
#[derive(Default, Debug)]
struct Node {
    edge: Vec<i32>,
    children: BTreeMap<i32, Box<Node>>,
    terminal: Option<u32>,
    representative: Option<u32>,
}
impl Node {
    fn refresh(&mut self) {
        self.representative = self
            .terminal
            .or_else(|| self.children.values().find_map(|node| node.representative));
    }
    fn insert(&mut self, key: &[i32], sequence: u32) {
        if key.is_empty() {
            self.terminal = Some(sequence);
            self.refresh();
            return;
        }
        let first = key[0];
        if let Some(mut child) = self.children.remove(&first) {
            let shared = key
                .iter()
                .zip(&child.edge)
                .take_while(|(a, b)| a == b)
                .count();
            if shared == child.edge.len() {
                child.insert(&key[shared..], sequence);
            } else {
                let mut parent = Node {
                    edge: child.edge[..shared].to_vec(),
                    ..Node::default()
                };
                child.edge = child.edge[shared..].to_vec();
                parent.children.insert(child.edge[0], child);
                parent.insert(&key[shared..], sequence);
                child = Box::new(parent);
            }
            self.children.insert(first, child);
        } else {
            self.children.insert(
                first,
                Box::new(Node {
                    edge: key.to_vec(),
                    terminal: Some(sequence),
                    representative: Some(sequence),
                    ..Node::default()
                }),
            );
        }
        self.refresh();
    }
    fn remove(&mut self, key: &[i32]) {
        if key.is_empty() {
            self.terminal = None;
            self.refresh();
            return;
        }
        let first = key[0];
        if let Some(mut child) = self.children.remove(&first) {
            debug_assert!(key.starts_with(&child.edge));
            child.remove(&key[child.edge.len()..]);
            if child.terminal.is_some() || !child.children.is_empty() {
                if child.terminal.is_none() && child.children.len() == 1 {
                    let (_, mut only) = child.children.pop_first().unwrap();
                    let mut edge = child.edge;
                    edge.append(&mut only.edge);
                    only.edge = edge;
                    child = only;
                }
                self.children.insert(first, child);
            }
        }
        self.refresh();
    }
    fn exact(&self, key: &[i32]) -> bool {
        if key.is_empty() {
            return self.terminal.is_some();
        }
        match self.children.get(&key[0]) {
            Some(child) if key.starts_with(&child.edge) => child.exact(&key[child.edge.len()..]),
            _ => false,
        }
    }
    fn node_count(&self) -> usize {
        1 + self
            .children
            .values()
            .map(|node| node.node_count())
            .sum::<usize>()
    }
}

#[derive(Default, Debug)]
pub struct PrefixCache {
    root: Node,
    entries: HashMap<u32, CacheEntry>,
    lru: BTreeSet<(u128, u32)>,
    clock: u128,
    tokens: usize,
}
impl PrefixCache {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Conservative count: sum of retained prompt lengths, including common prefixes.
    pub fn token_count(&self) -> usize {
        self.tokens
    }
    pub fn node_count(&self) -> usize {
        self.root.node_count()
    }
    pub fn contains(&self, tokens: &[i32]) -> bool {
        self.root.exact(tokens)
    }
    pub fn lookup(&mut self, tokens: &[i32]) -> Option<PrefixMatch> {
        let mut node = &self.root;
        let mut offset = 0;
        let mut sequence = None;
        while offset < tokens.len() {
            let Some(child) = node.children.get(&tokens[offset]) else {
                break;
            };
            let shared = tokens[offset..]
                .iter()
                .zip(&child.edge)
                .take_while(|(a, b)| a == b)
                .count();
            if shared == 0 {
                break;
            }
            offset += shared;
            sequence = child.representative;
            if shared != child.edge.len() {
                break;
            }
            node = child;
        }
        let sequence = sequence?;
        let entry = self
            .entries
            .get_mut(&sequence)
            .expect("radix representative must be live");
        self.lru.remove(&(entry.stamp, sequence));
        self.clock += 1;
        entry.stamp = self.clock;
        self.lru.insert((entry.stamp, sequence));
        Some(PrefixMatch {
            sequence,
            tokens: offset,
        })
    }
    /// The caller owns physical KV. Insert only after successfully copying to this sequence.
    pub fn insert(&mut self, tokens: Vec<i32>, sequence: u32) -> Result<()> {
        if tokens.is_empty() {
            return Err(Error::new("cannot cache an empty prompt"));
        }
        if self.entries.contains_key(&sequence) || self.contains(&tokens) {
            return Err(Error::new("duplicate cache sequence or prompt"));
        }
        self.tokens = self
            .tokens
            .checked_add(tokens.len())
            .ok_or_else(|| Error::new("cache token count overflow"))?;
        self.root.insert(&tokens, sequence);
        self.clock += 1;
        self.lru.insert((self.clock, sequence));
        self.entries.insert(
            sequence,
            CacheEntry {
                sequence,
                tokens,
                stamp: self.clock,
            },
        );
        Ok(())
    }
    pub fn remove(&mut self, sequence: u32) -> Option<CacheEntry> {
        let entry = self.entries.remove(&sequence)?;
        self.lru.remove(&(entry.stamp, sequence));
        self.tokens -= entry.tokens.len();
        self.root.remove(&entry.tokens);
        Some(entry)
    }
    pub fn pop_lru(&mut self) -> Option<CacheEntry> {
        let sequence = self.lru.first()?.1;
        self.remove(sequence)
    }
}
