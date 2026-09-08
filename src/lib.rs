//! 原生教学运行时：调度策略、前缀索引和采样留在 Rust，模型计算经 Engine 边界接入。
//! 单拥有者 scheduler 使 KV 生命周期可以直接阅读，不引入共享可变状态或隐式后台线程。
pub mod cache;
pub mod engine;
#[cfg(feature = "llama")]
pub mod llama;
pub mod sampling;
pub mod scheduler;
pub mod types;

pub use engine::{BatchToken, Engine};
pub use scheduler::Scheduler;
pub use types::*;
