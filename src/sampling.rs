//! 请求自带 RNG，避免并发批次的组成改变采样序列。CPU 实现直接展示稳定 softmax、
//! top-k 和 nucleus 的顺序；零温度以 O(V) argmax 完成，不排序或分配候选表。
use crate::types::{Error, Result, SamplingParams};
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}
fn descending(a: &(usize, f64), b: &(usize, f64)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
}
pub fn sample(logits: &[f32], params: &SamplingParams, rng: &mut Rng) -> Result<i32> {
    params.validate()?;
    if logits.is_empty() || logits.len() > i32::MAX as usize {
        return Err(Error::new("invalid logits vocabulary size"));
    }
    let mut best = None;
    for (token, value) in logits.iter().copied().enumerate() {
        if value.is_nan() || value == f32::INFINITY {
            return Err(Error::new("model returned non-finite logits"));
        }
        if value.is_finite() && best.is_none_or(|(_, previous)| value > previous) {
            best = Some((token, value));
        }
    }
    let (best_token, best_value) =
        best.ok_or_else(|| Error::new("model returned no finite logits"))?;
    if params.temperature == 0.0 {
        return Ok(best_token as i32);
    }
    let mut candidates: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, v)| (i, *v as f64))
        .collect();
    if params.top_k > 0 && params.top_k < candidates.len() {
        candidates.select_nth_unstable_by(params.top_k, descending);
        candidates.truncate(params.top_k);
    }
    // Nucleus needs descending probability order; unrestricted sampling needs no sort.
    if params.top_p < 1.0 {
        candidates.sort_unstable_by(descending);
    }
    let mut sum = 0.0;
    for (_, weight) in &mut candidates {
        *weight = ((*weight - best_value as f64) / params.temperature as f64).exp();
        sum += *weight;
    }
    let mut kept_sum = sum;
    if params.top_p < 1.0 {
        let threshold = sum * params.top_p as f64;
        kept_sum = 0.0;
        let mut count = 0;
        for (_, weight) in &candidates {
            kept_sum += *weight;
            count += 1;
            if kept_sum >= threshold {
                break;
            }
        }
        candidates.truncate(count);
    }
    let target = rng.uniform() * kept_sum;
    let mut cumulative = 0.0;
    for (token, weight) in &candidates {
        cumulative += *weight;
        if target < cumulative {
            return Ok(*token as i32);
        }
    }
    Ok(candidates.last().unwrap().0 as i32)
}
