//! 请求、事件和容量配置是运行时的公开协议。使用值类型明确所有权，让 CLI/HTTP
//! 只处理输入输出，不必触碰序列或 KV；错误通过 Result 返回而不是静默截断。
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub message: String,
}
impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// Zero means greedy; finite positive values scale logits.
    pub temperature: f32,
    /// Zero means the entire vocabulary.
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}
impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        }
    }
}
impl SamplingParams {
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(Error::new("temperature must be finite and non-negative"));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(Error::new("top_p must be finite and in (0, 1]"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct GenerationRequest {
    pub id: u64,
    pub prompt: String,
    pub max_new_tokens: usize,
    pub sampling: SamplingParams,
}

#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub max_active: usize,
    pub max_pending: usize,
    pub max_batch_tokens: usize,
    pub prefill_chunk_size: usize,
    /// Conservative total: active worst-case reservations plus cached sequence lengths.
    pub max_kv_tokens: usize,
    pub max_sequence_tokens: usize,
    /// Shared, bounded ID pool for active requests and cache entries.
    pub max_sequences: usize,
    pub max_cached_tokens: usize,
    pub max_cached_entries: usize,
}
impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_active: 8,
            max_pending: 128,
            max_batch_tokens: 512,
            prefill_chunk_size: 256,
            max_kv_tokens: 4096,
            max_sequence_tokens: 4096,
            max_sequences: 64,
            max_cached_tokens: 1024,
            max_cached_entries: 56,
        }
    }
}
impl SchedulerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_active == 0
            || self.max_pending == 0
            || self.max_batch_tokens == 0
            || self.prefill_chunk_size == 0
            || self.max_kv_tokens == 0
            || self.max_sequence_tokens == 0
        {
            return Err(Error::new(
                "active, pending, batch, chunk and KV capacities must be positive",
            ));
        }
        if self.max_active > self.max_batch_tokens {
            return Err(Error::new(
                "max_batch_tokens must cover one decode token per active request",
            ));
        }
        if self.max_sequences < self.max_active || self.max_sequences > i32::MAX as usize {
            return Err(Error::new(
                "max_sequences must cover max_active and fit llama sequence IDs",
            ));
        }
        if self.max_cached_tokens > self.max_kv_tokens {
            return Err(Error::new("max_cached_tokens cannot exceed max_kv_tokens"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Eos,
    Length,
    Cancelled,
}
impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Eos => "stop",
            Self::Length => "length",
            Self::Cancelled => "cancelled",
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// Raw token pieces may split a UTF-8 character: decode in the frontend across events.
    Token {
        id: u64,
        token: i32,
        bytes: Vec<u8>,
    },
    Finished {
        id: u64,
        reason: FinishReason,
        prompt_tokens: usize,
        generated_tokens: usize,
        cached_tokens: usize,
        prefill_tokens: usize,
    },
    Error {
        id: u64,
        message: String,
    },
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub pending: usize,
    pub active: usize,
    pub cached_entries: usize,
    pub cached_tokens: usize,
    pub reserved_tokens: usize,
    pub free_sequences: usize,
    pub total_submitted: u64,
    pub total_completed: u64,
    pub total_cancelled: u64,
    pub total_failed: u64,
    pub prefill_tokens: u64,
    pub decode_tokens: u64,
    pub cache_hit_tokens: u64,
    pub forward_batches: u64,
}
