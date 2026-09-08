//! 命令行只负责边界参数和展示；调度与推理留在库里，可独立测试与替换。
#[cfg(feature = "llama")]
use mini_sglang::types::SchedulerConfig;
use mini_sglang::types::{Error, Result, SamplingParams};
#[cfg(feature = "llama")]
use serde_json::{json, Value};

#[cfg(feature = "llama")]
pub const MAX_PENDING: usize = 64;

#[derive(Clone, Debug)]
pub struct Options {
    pub model: String,
    pub prompts: Vec<String>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub batch_tokens: usize,
    pub prefill_chunk: usize,
    pub max_sequences: usize,
    pub context_size: usize,
    pub cache_sequences: usize,
    pub threads: usize,
    pub gpu_layers: i32,
    pub json: bool,
    pub serve: bool,
    pub host: String,
    pub port: u16,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model: String::new(),
            prompts: Vec::new(),
            max_tokens: 32,
            sampling: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                seed: 0,
            },
            batch_tokens: 256,
            prefill_chunk: 64,
            max_sequences: 8,
            context_size: 4096,
            cache_sequences: 8,
            threads: 4,
            gpu_layers: 0,
            json: false,
            serve: false,
            host: "127.0.0.1".into(),
            port: 1919,
        }
    }
}

pub const HELP: &str = "mini-sglang (Rust) — native GGUF inference

Usage:
  mini-sglang --model model.gguf --prompt text [--prompt text ...] [options]
  mini-sglang --model model.gguf --serve [--host 127.0.0.1 --port 1919]

Options:
  --max-tokens N       Maximum generated tokens; 0 is allowed (default 32)
  --temperature F     Sampling temperature; 0 selects greedy (default 0)
  --top-p F           Nucleus probability in (0, 1] (default 1)
  --top-k N           Candidate limit; 0 disables (default 0)
  --seed N            Per-request RNG seed (default 0)
  --batch-tokens N     Shared forward token budget (default 256)
  --prefill-chunk N    Maximum prefill tokens per scheduling step (default 64)
  --max-sequences N    Concurrent active requests (default 8)
  --context-size N     Total shared KV token capacity (default 4096)
  --cache-sequences N  Maximum retained prefix sequences (default 8)
  --threads N          CPU inference threads (default 4)
  --gpu-layers N       Offloaded layers; nonnegative (default 0)
  --json               Print one JSON result object
  --serve              Start bounded HTTP server
  --host ADDRESS       Bind address (default 127.0.0.1)
  --port N             TCP port (default 1919)
  --help               Print this help

Inference requires a binary built with --features llama and a real GGUF model.
";

fn unsigned(value: &str, flag: &str) -> Result<usize> {
    if value.is_empty() || !value.bytes().all(|x| x.is_ascii_digit()) {
        return Err(Error::new(format!("{flag} requires a nonnegative integer")));
    }
    value
        .parse()
        .map_err(|_| Error::new(format!("{flag} is out of range")))
}

fn float(value: &str, flag: &str) -> Result<f32> {
    let number: f32 = value
        .parse()
        .map_err(|_| Error::new(format!("{flag} requires a number")))?;
    if !number.is_finite() {
        return Err(Error::new(format!("{flag} requires a finite number")));
    }
    Ok(number)
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Option<Options>> {
    let mut options = Options::default();
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--help" | "-h" => return Ok(None),
            "--json" => options.json = true,
            "--serve" => options.serve = true,
            "--model" | "--prompt" | "--max-tokens" | "--temperature" | "--top-p" | "--top-k"
            | "--seed" | "--batch-tokens" | "--prefill-chunk" | "--max-sequences"
            | "--context-size" | "--cache-sequences" | "--threads" | "--gpu-layers" | "--host"
            | "--port" => {
                let value = args
                    .next()
                    .ok_or_else(|| Error::new(format!("missing value for {flag}")))?;
                match flag.as_str() {
                    "--model" => options.model = value,
                    "--prompt" => options.prompts.push(value),
                    "--max-tokens" => options.max_tokens = unsigned(&value, &flag)?,
                    "--temperature" => options.sampling.temperature = float(&value, &flag)?,
                    "--top-p" => options.sampling.top_p = float(&value, &flag)?,
                    "--top-k" => options.sampling.top_k = unsigned(&value, &flag)?,
                    "--seed" => {
                        if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                            return Err(Error::new("--seed requires a nonnegative integer"));
                        }
                        options.sampling.seed = value
                            .parse()
                            .map_err(|_| Error::new("--seed is out of range"))?;
                    }
                    "--batch-tokens" => options.batch_tokens = unsigned(&value, &flag)?,
                    "--prefill-chunk" => options.prefill_chunk = unsigned(&value, &flag)?,
                    "--max-sequences" => options.max_sequences = unsigned(&value, &flag)?,
                    "--context-size" => options.context_size = unsigned(&value, &flag)?,
                    "--cache-sequences" => options.cache_sequences = unsigned(&value, &flag)?,
                    "--threads" => options.threads = unsigned(&value, &flag)?,
                    "--gpu-layers" => {
                        options.gpu_layers = value.parse().map_err(|_| {
                            Error::new("--gpu-layers requires a nonnegative integer")
                        })?
                    }
                    "--host" => options.host = value,
                    "--port" => {
                        let port = unsigned(&value, &flag)?;
                        options.port = u16::try_from(port)
                            .map_err(|_| Error::new("--port must be in 1..65535"))?;
                    }
                    _ => return Err(Error::new("unrecognized option")),
                }
            }
            _ => return Err(Error::new(format!("unknown argument: {flag}"))),
        }
    }
    options.validate()?;
    Ok(Some(options))
}

impl Options {
    fn validate(&self) -> Result<()> {
        if self.model.is_empty() {
            return Err(Error::new("--model is required; use --help for usage"));
        }
        if self.serve && !self.prompts.is_empty() {
            return Err(Error::new("--serve and --prompt cannot be combined"));
        }
        if !self.serve && self.prompts.is_empty() {
            return Err(Error::new("at least one --prompt or --serve is required"));
        }
        if self.prompts.iter().any(|p| p.is_empty()) {
            return Err(Error::new("prompt must not be empty"));
        }
        if self.max_tokens > i32::MAX as usize {
            return Err(Error::new("--max-tokens is out of range"));
        }
        validate_sampling(&self.sampling)?;
        for (name, value) in [
            ("--batch-tokens", self.batch_tokens),
            ("--prefill-chunk", self.prefill_chunk),
            ("--max-sequences", self.max_sequences),
            ("--context-size", self.context_size),
            ("--threads", self.threads),
        ] {
            if value == 0 || value > i32::MAX as usize {
                return Err(Error::new(format!("{name} must be in 1..2147483647")));
            }
        }
        if self.cache_sequences > i32::MAX as usize
            || self
                .max_sequences
                .checked_add(self.cache_sequences)
                .is_none_or(|v| v > i32::MAX as usize)
        {
            return Err(Error::new(
                "active plus cached sequence count is out of range",
            ));
        }
        if self.prefill_chunk > self.batch_tokens {
            return Err(Error::new("--prefill-chunk cannot exceed --batch-tokens"));
        }
        if self.max_sequences > self.batch_tokens {
            return Err(Error::new(
                "--batch-tokens must cover one decode token per active sequence",
            ));
        }
        if self.gpu_layers < 0 {
            return Err(Error::new("--gpu-layers must be nonnegative"));
        }
        if self.host.is_empty() || self.port == 0 {
            return Err(Error::new(
                "--host must not be empty and --port must be in 1..65535",
            ));
        }
        Ok(())
    }

    #[cfg(feature = "llama")]
    pub fn scheduler_config(&self) -> SchedulerConfig {
        SchedulerConfig {
            max_active: self.max_sequences,
            max_pending: MAX_PENDING,
            max_batch_tokens: self.batch_tokens,
            prefill_chunk_size: self.prefill_chunk,
            max_kv_tokens: self.context_size,
            max_sequence_tokens: self.context_size,
            max_sequences: self.max_sequences + self.cache_sequences,
            max_cached_tokens: if self.cache_sequences == 0 {
                0
            } else {
                self.context_size / 2
            },
            max_cached_entries: self.cache_sequences,
        }
    }

    #[cfg(feature = "llama")]
    pub fn engine(&self) -> Result<mini_sglang::llama::LlamaEngine> {
        mini_sglang::llama::LlamaEngine::new(mini_sglang::llama::LlamaConfig {
            model_path: self.model.clone(),
            context_tokens: self.context_size as u32,
            batch_size: self.batch_tokens as u32,
            max_sequences: (self.max_sequences + self.cache_sequences) as u32,
            threads: self.threads as i32,
            gpu_layers: self.gpu_layers,
        })
    }
}

pub fn validate_sampling(params: &SamplingParams) -> Result<()> {
    if !params.temperature.is_finite() || params.temperature < 0.0 {
        return Err(Error::new(
            "temperature must be a finite nonnegative number",
        ));
    }
    if !params.top_p.is_finite() || params.top_p <= 0.0 || params.top_p > 1.0 {
        return Err(Error::new("top_p must be in (0, 1]"));
    }
    if params.top_k > i32::MAX as usize {
        return Err(Error::new("top_k is out of range"));
    }
    Ok(())
}

/// token 可以从 UTF-8 字符中间切分；保留不完整尾部，避免逐 token 产生替代字符。
#[derive(Default)]
#[cfg(any(feature = "llama", test))]
pub struct Utf8Output {
    pending: Vec<u8>,
    pub text: String,
}

#[cfg(any(feature = "llama", test))]
impl Utf8Output {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut chunk = String::new();
        let mut consumed = 0;
        while consumed < self.pending.len() {
            match std::str::from_utf8(&self.pending[consumed..]) {
                Ok(valid) => {
                    chunk.push_str(valid);
                    consumed = self.pending.len();
                }
                Err(error) => {
                    let end = consumed + error.valid_up_to();
                    // valid_up_to 是标准库已验证的边界，不依赖模型输出的完整性。
                    if let Ok(valid) = std::str::from_utf8(&self.pending[consumed..end]) {
                        chunk.push_str(valid);
                    }
                    consumed = end;
                    if let Some(length) = error.error_len() {
                        chunk.push(char::REPLACEMENT_CHARACTER);
                        consumed += length;
                    } else {
                        break;
                    }
                }
            }
        }
        self.pending.drain(..consumed);
        self.text.push_str(&chunk);
        chunk
    }

    pub fn finish(&mut self) -> String {
        let tail = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        self.text.push_str(&tail);
        tail
    }
}

#[cfg(feature = "llama")]
pub fn usage(prompt: usize, completion: usize, cached: usize, prefill: usize) -> Value {
    json!({"prompt_tokens":prompt,"completion_tokens":completion,"total_tokens":prompt + completion,
        "cached_tokens":cached,"prefill_tokens":prefill})
}

#[cfg(feature = "llama")]
pub fn stats_json(stats: mini_sglang::types::Stats) -> Value {
    json!({
        "pending":stats.pending,"active":stats.active,"cached_entries":stats.cached_entries,
        "cached_tokens":stats.cached_tokens,"reserved_tokens":stats.reserved_tokens,
        "free_sequences":stats.free_sequences,"total_submitted":stats.total_submitted,
        "total_completed":stats.total_completed,"total_cancelled":stats.total_cancelled,
        "total_failed":stats.total_failed,"prefill_tokens":stats.prefill_tokens,
        "decode_tokens":stats.decode_tokens,"cache_hit_tokens":stats.cache_hit_tokens,
        "forward_batches":stats.forward_batches
    })
}

pub fn run() -> Result<()> {
    let Some(options) = parse_args(std::env::args().skip(1))? else {
        print!("{HELP}");
        return Ok(());
    };
    #[cfg(not(feature = "llama"))]
    {
        let _ = options;
        Err(Error::new("inference backend unavailable: rebuild with cargo build --release --features llama (see README)"))
    }
    #[cfg(feature = "llama")]
    {
        if options.serve {
            crate::server::serve(options)
        } else {
            run_cli(options)
        }
    }
}

#[cfg(feature = "llama")]
fn run_cli(options: Options) -> Result<()> {
    use mini_sglang::{
        scheduler::Scheduler,
        types::{Event, GenerationRequest},
    };
    let mut scheduler = Scheduler::new(options.engine()?, options.scheduler_config())?;
    let mut output: Vec<Utf8Output> = options
        .prompts
        .iter()
        .map(|_| Utf8Output::default())
        .collect();
    let mut tokens: Vec<Vec<i32>> = options.prompts.iter().map(|_| Vec::new()).collect();
    let mut results = vec![Value::Null; options.prompts.len()];
    let mut submitted = 0;
    while submitted < options.prompts.len() || !scheduler.idle() {
        while submitted < options.prompts.len() && scheduler.stats().pending < MAX_PENDING {
            scheduler.submit(GenerationRequest {
                id: submitted as u64 + 1,
                prompt: options.prompts[submitted].clone(),
                max_new_tokens: options.max_tokens,
                sampling: options.sampling.clone(),
            })?;
            submitted += 1;
        }
        for event in scheduler.step()? {
            match event {
                Event::Token { id, token, bytes } => {
                    let index = id as usize - 1;
                    output[index].push(&bytes);
                    tokens[index].push(token);
                }
                Event::Finished {
                    id,
                    reason,
                    prompt_tokens,
                    generated_tokens,
                    cached_tokens,
                    prefill_tokens,
                } => {
                    let index = id as usize - 1;
                    output[index].finish();
                    results[index] = json!({"request_id":id,"prompt":options.prompts[index],
                        "text":output[index].text,"token_ids":tokens[index],"finish_reason":reason.to_string(),
                        "usage":usage(prompt_tokens,generated_tokens,cached_tokens,prefill_tokens)});
                }
                Event::Error { id, message } => {
                    return Err(Error::new(format!("request {id}: {message}")))
                }
            }
        }
    }
    if options.json {
        println!(
            "{}",
            json!({"results":results,"stats":stats_json(scheduler.stats())})
        );
    } else {
        for result in results {
            if let Some(text) = result.get("text").and_then(Value::as_str) {
                println!("{text}");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_fragments_are_joined_across_tokens() {
        let mut output = Utf8Output::default();
        assert_eq!(output.push(&[0xe4]), "");
        assert_eq!(output.push(&[0xb8, 0xad, 0xf0, 0x9f]), "中");
        assert_eq!(output.push(&[0x8c, 0x8d]), "🌍");
        assert_eq!(output.finish(), "");
        assert_eq!(output.text, "中🌍");
    }

    #[test]
    fn invalid_utf8_does_not_discard_following_text() {
        let mut output = Utf8Output::default();
        assert_eq!(output.push(&[0xff, b'A', 0xe4]), "�A");
        assert_eq!(output.finish(), "�");
    }

    #[test]
    fn cli_rejects_negative_or_nonfinite_numbers() {
        for (flag, value) in [
            ("--max-tokens", "-1"),
            ("--temperature", "NaN"),
            ("--top-p", "0"),
            ("--port", "0"),
            ("--threads", "0"),
            ("--gpu-layers", "-1"),
        ] {
            let args = ["--model", "m.gguf", "--prompt", "a", flag, value].map(str::to_owned);
            assert!(parse_args(args).is_err(), "{flag} {value}");
        }
    }
}
