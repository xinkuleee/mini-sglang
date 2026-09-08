//! 单拥有者 scheduler 是请求与 KV 的唯一生命周期管理者。submit 只进入有界队列，
//! step 先安排每个 decode 的一个 token，再用剩余预算分块 prefill；Vec 保持请求顺序。
//! 入场即预留最坏 KV 用量，缓存可被驱逐但活动序列不可被抢占，因此无需复杂回滚策略。
//! 模型错误使整个实例失效并尽力清理，避免在未知的物理 KV 状态上继续生成。
use crate::cache::PrefixCache;
use crate::engine::{BatchToken, Engine};
use crate::sampling::{sample, Rng};
use crate::types::*;
use std::collections::{HashSet, VecDeque};

struct Pending {
    request: GenerationRequest,
    tokens: Vec<i32>,
    reservation: usize,
}
struct Active {
    pending: Pending,
    sequence: u32,
    processed: usize,
    generated: usize,
    last_token: Option<i32>,
    cache_hit: usize,
    prefill_tokens: usize,
    rng: Rng,
}
struct Plan {
    index: usize,
    count: usize,
    prefill: bool,
    logits: bool,
}

pub struct Scheduler<E: Engine> {
    engine: E,
    config: SchedulerConfig,
    pending: VecDeque<Pending>,
    active: Vec<Active>,
    ids: HashSet<u64>,
    cancelled: HashSet<u64>,
    cache: PrefixCache,
    free: Vec<u32>,
    used: Vec<bool>,
    reserved: usize,
    counters: Stats,
    poisoned: Option<Error>,
}
impl<E: Engine> Scheduler<E> {
    pub fn new(engine: E, config: SchedulerConfig) -> Result<Self> {
        config.validate()?;
        let free = (0..config.max_sequences as u32).rev().collect();
        let used = vec![false; config.max_sequences];
        Ok(Self {
            engine,
            config,
            pending: VecDeque::new(),
            active: Vec::new(),
            ids: HashSet::new(),
            cancelled: HashSet::new(),
            cache: PrefixCache::new(),
            free,
            used,
            reserved: 0,
            counters: Stats::default(),
            poisoned: None,
        })
    }
    pub fn engine(&self) -> &E {
        &self.engine
    }
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }
    pub fn healthy(&self) -> bool {
        self.poisoned.is_none()
    }
    fn check_health(&self) -> Result<()> {
        match &self.poisoned {
            Some(error) => Err(Error::new(format!("scheduler is unavailable: {error}"))),
            None => Ok(()),
        }
    }
    pub fn submit(&mut self, request: GenerationRequest) -> Result<()> {
        self.check_health()?;
        request.sampling.validate()?;
        if self.ids.contains(&request.id) {
            return Err(Error::new("duplicate live request ID"));
        }
        if self.pending.len() >= self.config.max_pending {
            return Err(Error::new("pending queue is full"));
        }
        // Check a cheap byte bound before tokenizer allocation. Ordinary tokenizer output is
        // bounded by input bytes plus special tokens; the exact token limit is checked below.
        if request.prompt.len() > self.config.max_sequence_tokens.saturating_mul(64) {
            return Err(Error::new("prompt byte length exceeds input limit"));
        }
        let tokens = self.engine.tokenize(&request.prompt)?;
        if tokens.is_empty() {
            return Err(Error::new(
                "tokenizer returned an empty prompt; a BOS token is required",
            ));
        }
        if tokens.iter().any(|token| *token < 0) {
            return Err(Error::new("tokenizer returned a negative token ID"));
        }
        let reservation = if request.max_new_tokens == 0 {
            0
        } else {
            tokens
                .len()
                .checked_add(request.max_new_tokens - 1)
                .ok_or_else(|| Error::new("requested sequence length overflow"))?
        };
        if tokens.len() > self.config.max_sequence_tokens
            || reservation > self.config.max_sequence_tokens
            || reservation > self.config.max_kv_tokens
        {
            return Err(Error::new(
                "request exceeds sequence or total KV token capacity",
            ));
        }
        self.ids.insert(request.id);
        self.pending.push_back(Pending {
            request,
            tokens,
            reservation,
        });
        self.counters.total_submitted += 1;
        Ok(())
    }
    /// Cancellation is applied at the next step, producing one terminal event.
    pub fn cancel(&mut self, id: u64) -> Result<()> {
        self.check_health()?;
        if !self.ids.contains(&id) {
            return Err(Error::new("unknown live request ID"));
        }
        self.cancelled.insert(id);
        Ok(())
    }
    pub fn idle(&self) -> bool {
        self.pending.is_empty() && self.active.is_empty()
    }
    pub fn stats(&self) -> Stats {
        let mut stats = self.counters.clone();
        stats.pending = self.pending.len();
        stats.active = self.active.len();
        stats.cached_entries = self.cache.len();
        stats.cached_tokens = self.cache.token_count();
        stats.reserved_tokens = self.reserved;
        stats.free_sequences = self.free.len();
        stats
    }
    pub fn step(&mut self) -> Result<Vec<Event>> {
        self.check_health()?;
        let mut events = Vec::new();
        if let Err(error) = self.step_inner(&mut events) {
            self.fail_all(error, &mut events);
        }
        Ok(events)
    }
    fn allocate(&mut self) -> u32 {
        let id = self.free.pop().expect("capacity checked before allocation");
        debug_assert!(!self.used[id as usize]);
        self.used[id as usize] = true;
        id
    }
    fn release(&mut self, sequence: u32) -> Result<()> {
        self.engine.remove_sequence(sequence)?;
        debug_assert!(self.used[sequence as usize]);
        self.used[sequence as usize] = false;
        self.free.push(sequence);
        Ok(())
    }
    fn evict_one(&mut self) -> Result<bool> {
        match self.cache.pop_lru() {
            Some(entry) => {
                self.release(entry.sequence)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }
    /// Active reservations are not evictable; waiting behind them preserves FIFO admission.
    fn make_room(&mut self, reservation: usize) -> Result<bool> {
        if reservation > self.config.max_kv_tokens - self.reserved {
            return Ok(false);
        }
        while self.cache.token_count() > self.config.max_kv_tokens - self.reserved - reservation
            || self.free.is_empty()
        {
            if !self.evict_one()? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn pending_finished(
        &mut self,
        pending: Pending,
        reason: FinishReason,
        events: &mut Vec<Event>,
    ) {
        self.ids.remove(&pending.request.id);
        self.cancelled.remove(&pending.request.id);
        if reason == FinishReason::Cancelled {
            self.counters.total_cancelled += 1;
        } else {
            self.counters.total_completed += 1;
        }
        events.push(Event::Finished {
            id: pending.request.id,
            reason,
            prompt_tokens: pending.tokens.len(),
            generated_tokens: 0,
            cached_tokens: 0,
            prefill_tokens: 0,
        });
    }
    fn finish(
        &mut self,
        index: usize,
        reason: FinishReason,
        events: &mut Vec<Event>,
    ) -> Result<()> {
        self.reserved -= self.active[index].pending.reservation;
        if reason != FinishReason::Cancelled {
            self.cache_prompt(index)?;
        }
        self.release(self.active[index].sequence)?;
        let active = self.active.remove(index);
        self.ids.remove(&active.pending.request.id);
        self.cancelled.remove(&active.pending.request.id);
        if reason == FinishReason::Cancelled {
            self.counters.total_cancelled += 1;
        } else {
            self.counters.total_completed += 1;
        }
        events.push(Event::Finished {
            id: active.pending.request.id,
            reason,
            prompt_tokens: active.pending.tokens.len(),
            generated_tokens: active.generated,
            cached_tokens: active.cache_hit,
            prefill_tokens: active.prefill_tokens,
        });
        Ok(())
    }
    fn step_inner(&mut self, events: &mut Vec<Event>) -> Result<()> {
        // Zero-output and cancelled requests consume no model work, even behind a blocked head.
        let count = self.pending.len();
        for _ in 0..count {
            let pending = self.pending.pop_front().unwrap();
            if self.cancelled.contains(&pending.request.id) {
                self.pending_finished(pending, FinishReason::Cancelled, events);
            } else if pending.request.max_new_tokens == 0 {
                self.pending_finished(pending, FinishReason::Length, events);
            } else {
                self.pending.push_back(pending);
            }
        }
        let mut index = 0;
        while index < self.active.len() {
            if self
                .cancelled
                .contains(&self.active[index].pending.request.id)
            {
                self.finish(index, FinishReason::Cancelled, events)?;
            } else {
                index += 1;
            }
        }
        while self.active.len() < self.config.max_active && !self.pending.is_empty() {
            let reservation = self.pending.front().unwrap().reservation;
            if !self.make_room(reservation)? {
                break;
            }
            let pending = self.pending.pop_front().unwrap();
            let sequence = self.allocate();
            let hit = self
                .cache
                .lookup(&pending.tokens[..pending.tokens.len() - 1]);
            let rng = Rng::new(pending.request.sampling.seed);
            self.reserved += pending.reservation;
            self.active.push(Active {
                pending,
                sequence,
                processed: 0,
                generated: 0,
                last_token: None,
                cache_hit: 0,
                prefill_tokens: 0,
                rng,
            });
            if let Some(hit) = hit {
                self.engine
                    .copy_sequence(hit.sequence, sequence, hit.tokens)?;
                let active = self.active.last_mut().unwrap();
                active.processed = hit.tokens;
                active.cache_hit = hit.tokens;
                self.counters.cache_hit_tokens += hit.tokens as u64;
            }
        }
        let mut batch = Vec::with_capacity(self.config.max_batch_tokens);
        let mut plans = Vec::new();
        // All decodes fit because max_active <= max_batch_tokens is a configuration invariant.
        for (index, active) in self.active.iter().enumerate() {
            if active.processed >= active.pending.tokens.len() {
                batch.push(BatchToken {
                    token: active.last_token.expect("decode has a sampled token"),
                    position: active.processed,
                    sequence: active.sequence,
                    logits: true,
                });
                plans.push(Plan {
                    index,
                    count: 1,
                    prefill: false,
                    logits: true,
                });
            }
        }
        // Reserve one token per prefill before distributing spare budget round-robin.
        // This prevents an earlier long prompt from starving later active prefills.
        let mut prefills: Vec<(usize, usize, usize)> = self
            .active
            .iter()
            .enumerate()
            .filter(|(_, active)| active.processed < active.pending.tokens.len())
            .map(|(index, active)| {
                (
                    index,
                    1,
                    (active.pending.tokens.len() - active.processed)
                        .min(self.config.prefill_chunk_size),
                )
            })
            .collect();
        let mut spare = self.config.max_batch_tokens - batch.len() - prefills.len();
        loop {
            let mut progress = false;
            for (_, count, limit) in &mut prefills {
                if spare > 0 && *count < *limit {
                    *count += 1;
                    spare -= 1;
                    progress = true;
                }
            }
            if !progress || spare == 0 {
                break;
            }
        }
        for (index, count, _) in prefills {
            let active = &self.active[index];
            let end = active.processed + count;
            let logits = end == active.pending.tokens.len();
            for position in active.processed..end {
                batch.push(BatchToken {
                    token: active.pending.tokens[position],
                    position,
                    sequence: active.sequence,
                    logits: logits && position + 1 == end,
                });
            }
            plans.push(Plan {
                index,
                count,
                prefill: true,
                logits,
            });
        }
        if batch.is_empty() {
            return Ok(());
        }
        let rows = self.engine.forward(&batch)?;
        self.counters.forward_batches += 1;
        if rows.len() != plans.iter().filter(|plan| plan.logits).count() {
            return Err(Error::new("model returned a wrong number of logits rows"));
        }
        for plan in &plans {
            let active = &mut self.active[plan.index];
            active.processed += plan.count;
            if plan.prefill {
                active.prefill_tokens += plan.count;
                self.counters.prefill_tokens += plan.count as u64;
            } else {
                self.counters.decode_tokens += plan.count as u64;
            }
        }
        let mut rows = rows.into_iter();
        let mut finished = Vec::new();
        for plan in plans {
            if !plan.logits {
                continue;
            }
            if plan.prefill {
                self.cache_prompt(plan.index)?;
            }
            let active = &mut self.active[plan.index];
            let token = sample(
                &rows.next().unwrap(),
                &active.pending.request.sampling,
                &mut active.rng,
            )?;
            active.generated += 1;
            active.last_token = Some(token);
            let id = active.pending.request.id;
            if self.engine.is_eog(token) {
                finished.push((id, FinishReason::Eos));
            } else {
                events.push(Event::Token {
                    id,
                    token,
                    bytes: self.engine.piece(token)?,
                });
                if active.generated == active.pending.request.max_new_tokens {
                    finished.push((id, FinishReason::Length));
                }
            }
        }
        for (id, reason) in finished {
            let index = self
                .active
                .iter()
                .position(|active| active.pending.request.id == id)
                .unwrap();
            self.finish(index, reason, events)?;
        }
        Ok(())
    }
    fn cache_prompt(&mut self, index: usize) -> Result<()> {
        let active = &mut self.active[index];
        let length = active.pending.tokens.len();
        if self.config.max_cached_entries == 0
            || length > self.config.max_cached_tokens
            || length > self.config.max_kv_tokens - self.reserved
            || self.cache.contains(&active.pending.tokens)
        {
            return Ok(());
        }
        while self.cache.len() >= self.config.max_cached_entries
            || self.cache.token_count() > self.config.max_cached_tokens - length
            || self.cache.token_count() > self.config.max_kv_tokens - self.reserved - length
            || self.free.is_empty()
        {
            if !self.evict_one()? {
                return Ok(());
            }
        }
        let destination = self.allocate();
        let active = &self.active[index];
        self.engine
            .copy_sequence(active.sequence, destination, length)?;
        self.cache
            .insert(active.pending.tokens.clone(), destination)?;
        Ok(())
    }
    fn fail_all(&mut self, error: Error, events: &mut Vec<Event>) {
        let mut message = error.to_string();
        // Include every allocated slot, including a destination of a failed partial copy.
        for sequence in 0..self.used.len() {
            if self.used[sequence] {
                match self.engine.remove_sequence(sequence as u32) {
                    Ok(()) => {
                        self.used[sequence] = false;
                        self.free.push(sequence as u32);
                    }
                    Err(cleanup) => {
                        message.push_str(&format!("; cleanup sequence {sequence}: {cleanup}"))
                    }
                }
            }
        }
        for active in self.active.drain(..) {
            events.push(Event::Error {
                id: active.pending.request.id,
                message: message.clone(),
            });
            self.counters.total_failed += 1;
        }
        for pending in self.pending.drain(..) {
            events.push(Event::Error {
                id: pending.request.id,
                message: message.clone(),
            });
            self.counters.total_failed += 1;
        }
        self.cache = PrefixCache::new();
        self.reserved = 0;
        self.ids.clear();
        self.cancelled.clear();
        self.poisoned = Some(Error::new(message));
    }
}
impl<E: Engine> Drop for Scheduler<E> {
    fn drop(&mut self) {
        for (sequence, used) in self.used.iter().enumerate() {
            if *used {
                let _ = self.engine.remove_sequence(sequence as u32);
            }
        }
    }
}
