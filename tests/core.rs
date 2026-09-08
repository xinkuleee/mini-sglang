//! 核心测试使用可观察的 Engine 替身验证 KV 和调度协议；生产路径始终需要真实模型。
use mini_sglang::cache::PrefixCache;
use mini_sglang::sampling::{sample, Rng};
use mini_sglang::{
    BatchToken, Engine, Error, Event, FinishReason, GenerationRequest, Result, SamplingParams,
    Scheduler, SchedulerConfig,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Default)]
struct State {
    sequences: HashMap<u32, Vec<i32>>,
    batches: Vec<Vec<BatchToken>>,
    copies: Vec<(u32, u32, usize)>,
    fail_forward: bool,
    fail_copy: bool,
    fail_remove: bool,
    wrong_rows: bool,
    invalid_logits: bool,
    eos: bool,
    pieces: bool,
}
#[derive(Clone)]
struct TestEngine(Rc<RefCell<State>>);
impl Engine for TestEngine {
    fn tokenize(&mut self, text: &str) -> Result<Vec<i32>> {
        Ok(text.bytes().map(|b| b as i32).collect())
    }
    fn piece(&self, token: i32) -> Result<Vec<u8>> {
        if self.0.borrow().pieces {
            Ok(vec![0xe4, token as u8])
        } else {
            Ok(vec![b'a' + token as u8])
        }
    }
    fn is_eog(&self, token: i32) -> bool {
        token == 0
    }
    fn forward(&mut self, batch: &[BatchToken]) -> Result<Vec<Vec<f32>>> {
        let mut state = self.0.borrow_mut();
        state.batches.push(batch.to_vec());
        let mut rows = Vec::new();
        for item in batch {
            let seq = state.sequences.entry(item.sequence).or_default();
            assert_eq!(
                seq.len(),
                item.position,
                "KV must be contiguous per sequence"
            );
            seq.push(item.token);
            let hash = seq.iter().fold(0u64, |h, token| {
                h.wrapping_mul(31).wrapping_add(*token as u64)
            });
            if item.logits {
                let mut logits = vec![f32::NEG_INFINITY; 17];
                for (i, logit) in logits.iter_mut().enumerate().skip(1) {
                    *logit = ((hash.wrapping_add(i as u64 * 7919)) % 101) as f32 / 60.0;
                }
                if state.eos {
                    logits[0] = 100.0;
                }
                if state.invalid_logits {
                    logits[2] = f32::NAN;
                }
                rows.push(logits);
            }
        }
        if state.fail_forward {
            return Err(Error::new(
                "test model forward failure after partial KV write",
            ));
        }
        if state.wrong_rows {
            rows.clear();
        }
        Ok(rows)
    }
    fn copy_sequence(&mut self, src: u32, dst: u32, end: usize) -> Result<()> {
        let mut state = self.0.borrow_mut();
        assert!(!state.sequences.contains_key(&dst));
        let tokens = state.sequences.get(&src).expect("source is alive")[..end].to_vec();
        state.sequences.insert(dst, tokens);
        state.copies.push((src, dst, end));
        if state.fail_copy {
            return Err(Error::new("test partial copy failure"));
        }
        Ok(())
    }
    fn remove_sequence(&mut self, sequence: u32) -> Result<()> {
        let mut state = self.0.borrow_mut();
        if state.fail_remove {
            return Err(Error::new("test remove failure"));
        }
        state.sequences.remove(&sequence);
        Ok(())
    }
}
fn config() -> SchedulerConfig {
    SchedulerConfig {
        max_active: 3,
        max_pending: 8,
        max_batch_tokens: 4,
        prefill_chunk_size: 2,
        max_kv_tokens: 128,
        max_sequence_tokens: 64,
        max_sequences: 8,
        max_cached_tokens: 64,
        max_cached_entries: 4,
    }
}
fn setup(config: SchedulerConfig) -> (Scheduler<TestEngine>, Rc<RefCell<State>>) {
    let state = Rc::new(RefCell::new(State::default()));
    (
        Scheduler::new(TestEngine(state.clone()), config).unwrap(),
        state,
    )
}
fn request(id: u64, prompt: &str, max_new_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        id,
        prompt: prompt.into(),
        max_new_tokens,
        sampling: SamplingParams::default(),
    }
}
fn drain(scheduler: &mut Scheduler<TestEngine>) -> Vec<Event> {
    let mut events = Vec::new();
    for _ in 0..1000 {
        if scheduler.idle() {
            return events;
        }
        events.extend(scheduler.step().unwrap());
    }
    panic!("scheduler did not make progress");
}
fn tokens(events: &[Event], id: u64) -> Vec<i32> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Token {
                id: found, token, ..
            } if *found == id => Some(*token),
            _ => None,
        })
        .collect()
}
#[test]
fn radix_compresses_splits_matches_partial_edges_and_merges_on_eviction() {
    let mut cache = PrefixCache::new();
    cache.insert(vec![1, 2, 3, 4], 10).unwrap();
    assert_eq!(cache.node_count(), 2);
    cache.insert(vec![1, 2, 7], 11).unwrap();
    assert_eq!(cache.node_count(), 4);
    assert_eq!(cache.lookup(&[1, 2, 3, 9]).unwrap().tokens, 3);
    assert_eq!(cache.lookup(&[1, 2, 7, 8]).unwrap().sequence, 11);
    cache.insert(vec![1, 2], 12).unwrap();
    assert_eq!(cache.node_count(), 4);
    assert_eq!(cache.token_count(), 9);
    assert!(cache.insert(vec![1, 2], 13).is_err());
    assert!(cache.insert(vec![9], 10).is_err());
    cache.remove(12).unwrap();
    assert_eq!(cache.pop_lru().unwrap().sequence, 10);
    assert_eq!(cache.node_count(), 2);
    assert_eq!(cache.lookup(&[1, 2, 7]).unwrap().sequence, 11);
    cache.pop_lru().unwrap();
    assert!(cache.is_empty());
    assert_eq!(cache.node_count(), 1);
    assert_eq!(cache.token_count(), 0);
}
#[test]
fn radix_terminal_and_descendant_removal_never_leave_dangling_representatives() {
    let mut cache = PrefixCache::new();
    for i in 1..32u32 {
        cache.insert((0..i as i32).collect(), i).unwrap();
    }
    for i in (1..32u32).rev() {
        assert_eq!(
            cache
                .lookup(&(0..i as i32 + 1).collect::<Vec<_>>())
                .unwrap()
                .tokens,
            i as usize
        );
        cache.remove(i).unwrap();
    }
    assert!(cache.lookup(&[0]).is_none());
}
#[test]
fn chunked_prefill_mixes_with_decode_first_and_continuous_admission() {
    let (mut scheduler, state) = setup(config());
    scheduler.submit(request(1, "abcdef", 3)).unwrap();
    scheduler.submit(request(2, "xy", 3)).unwrap();
    let first = scheduler.step().unwrap();
    assert!(tokens(&first, 1).is_empty());
    assert_eq!(tokens(&first, 2).len(), 1);
    let short_sequence = state.borrow().batches[0][2].sequence;
    scheduler.step().unwrap();
    assert_eq!(state.borrow().batches[1][0].sequence, short_sequence);
    assert_eq!(state.borrow().batches[1][0].position, 2);
    scheduler.submit(request(3, "z", 1)).unwrap();
    let events = drain(&mut scheduler);
    assert_eq!(tokens(&events, 3).len(), 1);
    assert!(state.borrow().batches.iter().all(|batch| batch.len() <= 4));
    assert_eq!(scheduler.stats().total_completed, 3);
    assert_eq!(scheduler.stats().reserved_tokens, 0);
    drop(scheduler);
    assert!(state.borrow().sequences.is_empty());
}
#[test]
fn cached_prompt_recomputes_last_token_and_retains_no_generation_suffix() {
    let (mut scheduler, state) = setup(config());
    scheduler.submit(request(1, "abcd", 3)).unwrap();
    let first = drain(&mut scheduler);
    assert_eq!(scheduler.stats().cached_tokens, 4);
    let before = scheduler.stats();
    scheduler.submit(request(2, "abcd", 3)).unwrap();
    let second = drain(&mut scheduler);
    assert_eq!(tokens(&first, 1), tokens(&second, 2));
    assert_eq!(scheduler.stats().prefill_tokens - before.prefill_tokens, 1);
    assert_eq!(
        scheduler.stats().cache_hit_tokens - before.cache_hit_tokens,
        3
    );
    assert!(second.iter().any(|event| matches!(
        event,
        Event::Finished {
            cached_tokens: 3,
            prefill_tokens: 1,
            ..
        }
    )));
    assert_eq!(scheduler.stats().cached_tokens, 4);
    assert_eq!(state.borrow().sequences.len(), 1);
    assert_eq!(
        state.borrow().sequences.values().next().unwrap(),
        &vec![97, 98, 99, 100]
    );
}
#[test]
fn zero_output_and_pending_cancel_do_not_forward_or_allocate() {
    let (mut scheduler, state) = setup(config());
    scheduler.submit(request(1, "abc", 0)).unwrap();
    scheduler.submit(request(2, "abc", 4)).unwrap();
    scheduler.cancel(2).unwrap();
    let events = drain(&mut scheduler);
    assert!(state.borrow().batches.is_empty());
    assert!(state.borrow().sequences.is_empty());
    assert!(events.iter().any(|event| matches!(
        event,
        Event::Finished {
            id: 1,
            reason: FinishReason::Length,
            prefill_tokens: 0,
            generated_tokens: 0,
            ..
        }
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        Event::Finished {
            id: 2,
            reason: FinishReason::Cancelled,
            ..
        }
    )));
    assert_eq!(scheduler.stats().free_sequences, 8);
}
#[test]
fn active_cancel_eos_and_length_release_all_request_resources() {
    let mut cfg = config();
    cfg.max_cached_entries = 0;
    let (mut scheduler, state) = setup(cfg);
    scheduler.submit(request(1, "abcdef", 8)).unwrap();
    scheduler.step().unwrap();
    scheduler.cancel(1).unwrap();
    assert!(scheduler.step().unwrap().iter().any(|event| matches!(
        event,
        Event::Finished {
            reason: FinishReason::Cancelled,
            ..
        }
    )));
    assert!(state.borrow().sequences.is_empty());
    state.borrow_mut().eos = true;
    scheduler.submit(request(2, "a", 8)).unwrap();
    let events = drain(&mut scheduler);
    assert!(tokens(&events, 2).is_empty());
    assert!(events.iter().any(|event| matches!(
        event,
        Event::Finished {
            reason: FinishReason::Eos,
            generated_tokens: 1,
            ..
        }
    )));
    state.borrow_mut().eos = false;
    scheduler.submit(request(3, "a", 2)).unwrap();
    assert_eq!(tokens(&drain(&mut scheduler), 3).len(), 2);
    assert_eq!(scheduler.stats().free_sequences, 8);
    assert_eq!(scheduler.stats().reserved_tokens, 0);
    assert!(state.borrow().sequences.is_empty());
}
#[test]
fn admission_reserves_worst_case_kv_and_cache_eviction_is_bounded() {
    let mut cfg = config();
    cfg.max_kv_tokens = 8;
    cfg.max_cached_tokens = 4;
    cfg.max_cached_entries = 1;
    cfg.max_sequences = 3;
    let (mut scheduler, state) = setup(cfg);
    scheduler.submit(request(1, "abcd", 4)).unwrap();
    scheduler.submit(request(2, "xy", 3)).unwrap();
    scheduler.step().unwrap();
    assert_eq!(scheduler.stats().active, 1);
    assert_eq!(scheduler.stats().pending, 1);
    assert_eq!(scheduler.stats().reserved_tokens, 7);
    drain(&mut scheduler);
    for id in 3..20 {
        scheduler.submit(request(id, "qqq", 1)).unwrap();
        drain(&mut scheduler);
        let stats = scheduler.stats();
        assert!(stats.cached_entries <= 1);
        assert!(stats.reserved_tokens + stats.cached_tokens <= 8);
    }
    assert_eq!(
        scheduler.stats().free_sequences + scheduler.stats().cached_entries,
        3
    );
    drop(scheduler);
    assert!(state.borrow().sequences.is_empty());
}
#[test]
fn invalid_options_duplicate_ids_and_queue_pressure_are_rejected_before_work() {
    let mut cfg = config();
    cfg.max_pending = 1;
    let (mut scheduler, state) = setup(cfg);
    let mut invalid = request(1, "a", 1);
    invalid.sampling.temperature = f32::NAN;
    assert!(scheduler.submit(invalid).is_err());
    let mut invalid = request(1, "a", 1);
    invalid.sampling.top_p = 0.0;
    assert!(scheduler.submit(invalid).is_err());
    assert!(scheduler.submit(request(1, "", 1)).is_err());
    assert!(scheduler.submit(request(1, "a", usize::MAX)).is_err());
    scheduler.submit(request(1, "abc", 1)).unwrap();
    assert!(scheduler.submit(request(1, "abc", 1)).is_err());
    assert!(scheduler.submit(request(2, "abc", 1)).is_err());
    assert!(state.borrow().batches.is_empty());
    drain(&mut scheduler);
    scheduler.submit(request(1, "abc", 1)).unwrap();
}
#[test]
fn model_and_protocol_errors_poison_and_cleanup_every_sequence() {
    for kind in 0..4 {
        let (mut scheduler, state) = setup(config());
        match kind {
            0 => state.borrow_mut().fail_forward = true,
            1 => state.borrow_mut().wrong_rows = true,
            2 => state.borrow_mut().invalid_logits = true,
            _ => state.borrow_mut().fail_copy = true,
        }
        scheduler.submit(request(1, "ab", 3)).unwrap();
        scheduler.submit(request(2, "cd", 3)).unwrap();
        let events = drain(&mut scheduler);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::Error { .. }))
                .count(),
            2
        );
        assert!(!scheduler.healthy());
        assert!(scheduler.submit(request(3, "a", 1)).is_err());
        assert!(scheduler.step().is_err());
        assert!(state.borrow().sequences.is_empty());
        assert_eq!(scheduler.stats().free_sequences, 8);
        assert_eq!(scheduler.stats().reserved_tokens, 0);
    }
}
#[test]
fn cleanup_failure_is_reported_and_never_reuses_uncertain_sequence() {
    let (mut scheduler, state) = setup(config());
    scheduler.submit(request(1, "a", 1)).unwrap();
    state.borrow_mut().fail_remove = true;
    let events = scheduler.step().unwrap();
    assert!(events.iter().any(|event| matches!(event, Event::Error { message, .. } if message.contains("cleanup sequence"))));
    assert!(!scheduler.healthy());
    assert!(scheduler.stats().free_sequences < 8);
    state.borrow_mut().fail_remove = false;
    drop(scheduler);
    assert!(state.borrow().sequences.is_empty());
}
#[test]
fn stochastic_sampling_is_request_local_across_batch_composition_and_cache_hits() {
    let mut wanted = request(1, "abcdef", 12);
    wanted.sampling = SamplingParams {
        temperature: 0.8,
        top_k: 8,
        top_p: 0.85,
        seed: 123456,
    };
    let (mut solo, _) = setup(config());
    solo.submit(wanted.clone()).unwrap();
    let alone = tokens(&drain(&mut solo), 1);
    let (mut mixed, _) = setup(config());
    mixed.submit(request(9, "abcdef", 1)).unwrap();
    drain(&mut mixed);
    mixed.submit(request(2, "xy", 9)).unwrap();
    mixed.submit(wanted).unwrap();
    mixed
        .submit(request(3, "very long prompt here", 2))
        .unwrap();
    assert_eq!(alone, tokens(&drain(&mut mixed), 1));
}
#[test]
fn sampler_validates_logits_handles_ties_extremes_and_nucleus() {
    let mut rng = Rng::new(0);
    let greedy = SamplingParams::default();
    assert_eq!(sample(&[1.0, 1.0, 0.0], &greedy, &mut rng).unwrap(), 0);
    assert!(sample(&[], &greedy, &mut rng).is_err());
    assert!(sample(&[f32::NAN], &greedy, &mut rng).is_err());
    assert!(sample(&[f32::INFINITY], &greedy, &mut rng).is_err());
    assert!(sample(&[f32::NEG_INFINITY], &greedy, &mut rng).is_err());
    let top1 = SamplingParams {
        temperature: f32::MIN_POSITIVE,
        top_k: 1,
        top_p: 1.0,
        seed: 0,
    };
    assert_eq!(sample(&[-1e30, 1e30], &top1, &mut rng).unwrap(), 1);
    let nucleus = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 0.1,
        seed: 0,
    };
    for _ in 0..30 {
        assert_eq!(sample(&[0.0, 2.0, 1.0], &nucleus, &mut rng).unwrap(), 1);
    }
}
#[test]
fn token_events_preserve_raw_non_utf8_fragments() {
    let (mut scheduler, state) = setup(config());
    state.borrow_mut().pieces = true;
    scheduler.submit(request(1, "x", 1)).unwrap();
    assert!(drain(&mut scheduler).iter().any(|event| matches!(event, Event::Token { bytes, .. } if bytes[0] == 0xe4 && std::str::from_utf8(bytes).is_err())));
}

#[test]
fn full_reservation_can_cache_prompt_after_completion() {
    let mut cfg = config();
    cfg.max_kv_tokens = 6;
    cfg.max_cached_tokens = 6;
    let (mut scheduler, _) = setup(cfg);
    scheduler.submit(request(1, "abcd", 3)).unwrap();
    drain(&mut scheduler);
    assert_eq!(scheduler.stats().cached_tokens, 4);
    // A smaller output reservation permits cache and request to coexist at admission.
    scheduler.submit(request(2, "abcd", 1)).unwrap();
    // Total 4+4 exceeds six so admission must evict. A capacity eight variant does hit.
    drain(&mut scheduler);
    assert_eq!(scheduler.stats().cached_tokens, 4);
    let mut cfg = config();
    cfg.max_kv_tokens = 8;
    cfg.max_cached_tokens = 8;
    let (mut scheduler, _) = setup(cfg);
    scheduler.submit(request(1, "abcd", 5)).unwrap();
    drain(&mut scheduler);
    assert_eq!(scheduler.stats().cached_tokens, 4);
    scheduler.submit(request(2, "abcd", 1)).unwrap();
    assert!(drain(&mut scheduler).iter().any(|event| matches!(
        event,
        Event::Finished {
            cached_tokens: 3,
            prefill_tokens: 1,
            ..
        }
    )));
}
#[test]
fn every_active_prefill_progresses_under_a_tight_token_budget() {
    let mut cfg = config();
    cfg.max_batch_tokens = 3;
    cfg.prefill_chunk_size = 100;
    let (mut scheduler, state) = setup(cfg);
    for id in 1..=3 {
        scheduler.submit(request(id, "abcdefgh", 1)).unwrap();
    }
    scheduler.step().unwrap();
    let state = state.borrow();
    let batch = &state.batches[0];
    assert_eq!(batch.len(), 3);
    assert_eq!(
        batch
            .iter()
            .map(|item| item.sequence)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3
    );
    assert!(batch.iter().all(|item| item.position == 0));
}
