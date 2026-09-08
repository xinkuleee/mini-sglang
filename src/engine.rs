//! 模型边界只负责分词、前向计算和物理 KV 操作；不决定请求何时执行或如何采样。
//! 这样可用测试引擎验证策略，生产使用 llama.cpp 的真实权重与内核，而不依赖 Python。
use crate::types::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchToken {
    pub token: i32,
    pub position: usize,
    pub sequence: u32,
    pub logits: bool,
}

pub trait Engine {
    fn tokenize(&mut self, text: &str) -> Result<Vec<i32>>;
    fn piece(&self, token: i32) -> Result<Vec<u8>>;
    fn is_eog(&self, token: i32) -> bool;
    /// Return rows only for logits=true entries, in exactly their input order.
    fn forward(&mut self, batch: &[BatchToken]) -> Result<Vec<Vec<f32>>>;
    /// Copy [0, end_exclusive) into an empty destination; source remains valid.
    fn copy_sequence(&mut self, src: u32, dst: u32, end_exclusive: usize) -> Result<()>;
    /// Remove all KV references for this sequence. Must be safe for an empty sequence.
    fn remove_sequence(&mut self, id: u32) -> Result<()>;
}
