//! 固定 worker 池只做 HTTP；唯一模型线程持有 scheduler/context，避免 KV 操作并发。
//! 所有跨线程队列有上限，慢客户端取消请求，不能把模型内存变成无界输出缓冲。
use crate::frontend::{usage, validate_sampling, Options, Utf8Output, MAX_PENDING};
use mini_sglang::{
    types::{Error, Event, GenerationRequest, Result, SamplingParams, Stats},
    Scheduler,
};
use serde_json::{json, Map, Value};
use std::{
    collections::HashMap,
    io::{self, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError},
        Arc, Mutex,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const WORKERS: usize = 8;
const SOCKET_QUEUE: usize = 32;
const EVENT_QUEUE: usize = 256;
const MAX_HEADERS: usize = 16 * 1024;
const MAX_BODY: usize = 1024 * 1024;

#[derive(Clone, Debug)]
enum Prompt {
    Text(String),
    Chat(Vec<(String, String)>),
}

#[derive(Clone, Debug)]
struct Input {
    prompt: Prompt,
    max_tokens: usize,
    sampling: SamplingParams,
    stream: bool,
}

struct Command {
    id: u64,
    input: Input,
    accepted: SyncSender<std::result::Result<(), String>>,
    events: SyncSender<Event>,
    cancelled: Arc<AtomicBool>,
}

struct Subscription {
    events: SyncSender<Event>,
    cancelled: Arc<AtomicBool>,
}

// 即使 worker 因 I/O 提前返回，Drop 仍会通知 owner 回收请求及其 KV。
struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

pub fn serve(options: Options) -> Result<()> {
    let listener = TcpListener::bind((options.host.as_str(), options.port))
        .map_err(|e| Error::new(format!("cannot bind HTTP listener: {e}")))?;
    let (commands_tx, commands_rx) = mpsc::sync_channel::<Command>(MAX_PENDING);
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let healthy = Arc::new(AtomicBool::new(true));
    let owner_health = Arc::clone(&healthy);
    let snapshot = Arc::new(Mutex::new(Stats::default()));
    let owner_snapshot = Arc::clone(&snapshot);
    let owner_options = options.clone();
    thread::Builder::new()
        .name("model-owner".into())
        .spawn(move || {
            let result = owner_options
                .engine()
                .and_then(|engine| Scheduler::new(engine, owner_options.scheduler_config()));
            match result {
                Ok(scheduler) => {
                    if let Ok(mut snapshot) = owner_snapshot.lock() {
                        *snapshot = scheduler.stats();
                    }
                    let _ = ready_tx.send(Ok(()));
                    owner_loop(scheduler, commands_rx, owner_health, owner_snapshot);
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                }
            }
        })
        .map_err(|e| Error::new(format!("cannot start model owner: {e}")))?;
    ready_rx
        .recv()
        .map_err(|_| Error::new("model owner exited during startup"))??;

    let (sockets_tx, sockets_rx) = mpsc::sync_channel::<TcpStream>(SOCKET_QUEUE);
    let sockets_rx = Arc::new(Mutex::new(sockets_rx));
    let ids = Arc::new(AtomicU64::new(1));
    let options = Arc::new(options);
    for index in 0..WORKERS {
        let sockets = Arc::clone(&sockets_rx);
        let commands = commands_tx.clone();
        let ids = Arc::clone(&ids);
        let options = Arc::clone(&options);
        let healthy = Arc::clone(&healthy);
        let snapshot = Arc::clone(&snapshot);
        thread::Builder::new()
            .name(format!("http-{index}"))
            .spawn(move || loop {
                let socket = match sockets.lock() {
                    Ok(receiver) => receiver.recv(),
                    Err(_) => return,
                };
                let Ok(mut socket) = socket else {
                    return;
                };
                let _ = socket.set_read_timeout(Some(Duration::from_secs(5)));
                let _ = socket.set_write_timeout(Some(Duration::from_secs(5)));
                let _ = socket.set_nodelay(true);
                if let Err(error) =
                    handle(&mut socket, &commands, &ids, &options, &healthy, &snapshot)
                {
                    // 连接中断是正常取消，不重试已经写入的 HTTP 响应。
                    if !matches!(
                        error.kind(),
                        io::ErrorKind::BrokenPipe
                            | io::ErrorKind::ConnectionReset
                            | io::ErrorKind::UnexpectedEof
                    ) {
                        eprintln!("HTTP connection: {error}");
                    }
                }
            })
            .map_err(|e| Error::new(format!("cannot start HTTP worker: {e}")))?;
    }
    eprintln!(
        "mini-sglang listening on http://{}:{}",
        options.host, options.port
    );
    for connection in listener.incoming() {
        match connection {
            Ok(mut socket) => match sockets_tx.try_send(socket) {
                Ok(()) => {}
                Err(TrySendError::Full(returned)) => {
                    socket = returned;
                    let _ = socket.set_write_timeout(Some(Duration::from_millis(250)));
                    let _ = error_response(
                        &mut socket,
                        503,
                        "HTTP connection queue is full",
                        "overloaded",
                    );
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(Error::new("HTTP worker pool stopped"))
                }
            },
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(Error::new(format!("HTTP accept failed: {error}"))),
        }
    }
    Ok(())
}

fn owner_loop(
    mut scheduler: Scheduler<mini_sglang::llama::LlamaEngine>,
    commands: Receiver<Command>,
    healthy: Arc<AtomicBool>,
    snapshot: Arc<Mutex<Stats>>,
) {
    let mut subscriptions: HashMap<u64, Subscription> = HashMap::new();
    loop {
        if scheduler.idle() && subscriptions.is_empty() {
            match commands.recv() {
                Ok(command) => accept(&mut scheduler, &mut subscriptions, command),
                Err(_) => return,
            }
        }
        // 每轮有界接收，繁忙的 HTTP 队列不能饿死正在生成的请求。
        for _ in 0..MAX_PENDING {
            match commands.try_recv() {
                Ok(command) => accept(&mut scheduler, &mut subscriptions, command),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        let cancelled: Vec<u64> = subscriptions
            .iter()
            .filter_map(|(&id, sub)| sub.cancelled.load(Ordering::Acquire).then_some(id))
            .collect();
        for id in cancelled {
            subscriptions.remove(&id);
            let _ = scheduler.cancel(id);
        }
        match scheduler.step() {
            Ok(events) => {
                // 快照先于完成事件发布，客户端收到响应后读 health 不会看到旧计数。
                healthy.store(scheduler.healthy(), Ordering::Release);
                if let Ok(mut snapshot) = snapshot.lock() {
                    *snapshot = scheduler.stats();
                }
                for event in events {
                    let (id, terminal) = match &event {
                        Event::Token { id, .. } => (*id, false),
                        Event::Finished { id, .. } | Event::Error { id, .. } => (*id, true),
                    };
                    if let Some(sub) = subscriptions.get(&id) {
                        if sub.events.try_send(event).is_err() {
                            sub.cancelled.store(true, Ordering::Release);
                            subscriptions.remove(&id);
                            let _ = scheduler.cancel(id);
                        } else if terminal {
                            subscriptions.remove(&id);
                        }
                    }
                }
            }
            Err(error) => {
                for (id, sub) in subscriptions.drain() {
                    let _ = sub.events.try_send(Event::Error {
                        id,
                        message: error.to_string(),
                    });
                }
                // 核心把失败后端标为不可再用；新请求仍得到明确失败，进程不 panic。
            }
        }
        healthy.store(scheduler.healthy(), Ordering::Release);
        if let Ok(mut snapshot) = snapshot.lock() {
            *snapshot = scheduler.stats();
        }
    }
}

fn accept(
    scheduler: &mut Scheduler<mini_sglang::llama::LlamaEngine>,
    subscriptions: &mut HashMap<u64, Subscription>,
    command: Command,
) {
    if command.cancelled.load(Ordering::Acquire) {
        return;
    }
    let prompt = match &command.input.prompt {
        Prompt::Text(text) => Ok(text.clone()),
        Prompt::Chat(messages) => {
            let borrowed: Vec<(&str, &str)> = messages
                .iter()
                .map(|(role, content)| (role.as_str(), content.as_str()))
                .collect();
            scheduler.engine().apply_chat_template(&borrowed, true)
        }
    };
    let result = prompt.and_then(|prompt| {
        scheduler.submit(GenerationRequest {
            id: command.id,
            prompt,
            max_new_tokens: command.input.max_tokens,
            sampling: command.input.sampling,
        })
    });
    match result {
        Ok(()) => {
            if command.accepted.send(Ok(())).is_ok() {
                subscriptions.insert(
                    command.id,
                    Subscription {
                        events: command.events,
                        cancelled: command.cancelled,
                    },
                );
            } else {
                let _ = scheduler.cancel(command.id);
            }
        }
        Err(error) => {
            let _ = command.accepted.send(Err(error.to_string()));
        }
    }
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(socket: &mut TcpStream) -> std::result::Result<HttpRequest, (u16, String)> {
    let mut bytes = Vec::new();
    let header_end = loop {
        if let Some(position) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            if position + 4 > MAX_HEADERS {
                return Err((431, "request headers are too large".into()));
            }
            break position + 4;
        }
        if bytes.len() >= MAX_HEADERS {
            return Err((431, "request headers are too large".into()));
        }
        let mut buffer = [0u8; 2048];
        let count = socket
            .read(&mut buffer)
            .map_err(|e| (400, format!("cannot read request: {e}")))?;
        if count == 0 {
            return Err((400, "incomplete request headers".into()));
        }
        bytes.extend_from_slice(&buffer[..count]);
    };
    let header = std::str::from_utf8(&bytes[..header_end])
        .map_err(|_| (400, "headers must be UTF-8/ASCII".into()))?;
    let mut lines = header.split("\r\n");
    let first: Vec<&str> = lines
        .next()
        .unwrap_or_default()
        .split_whitespace()
        .collect();
    if first.len() != 3 || !matches!(first[2], "HTTP/1.0" | "HTTP/1.1") {
        return Err((400, "invalid HTTP request line".into()));
    }
    let method = first[0].to_owned();
    let path = first[1].to_owned();
    let mut content_length = None;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or((400, "invalid HTTP header".into()))?;
        let value = value.trim();
        if name.eq_ignore_ascii_case("transfer-encoding") {
            return Err((
                400,
                "Transfer-Encoding is unsupported; send Content-Length".into(),
            ));
        }
        if name.eq_ignore_ascii_case("expect") {
            return Err((417, "Expect is unsupported".into()));
        }
        if name.eq_ignore_ascii_case("content-length") {
            if content_length.is_some()
                || value.is_empty()
                || !value.bytes().all(|b| b.is_ascii_digit())
            {
                return Err((400, "invalid or duplicate Content-Length".into()));
            }
            content_length = Some(
                value
                    .parse::<usize>()
                    .map_err(|_| (413, "request body is too large".into()))?,
            );
        }
    }
    if method == "POST" && content_length.is_none() {
        return Err((411, "Content-Length is required".into()));
    }
    let length = content_length.unwrap_or(0);
    if length > MAX_BODY {
        return Err((413, "request body exceeds 1 MiB".into()));
    }
    let mut body = bytes.split_off(header_end);
    body.truncate(length);
    while body.len() < length {
        let mut buffer = [0u8; 4096];
        let wanted = buffer.len().min(length - body.len());
        let count = socket
            .read(&mut buffer[..wanted])
            .map_err(|e| (400, format!("cannot read request body: {e}")))?;
        if count == 0 {
            return Err((400, "incomplete request body".into()));
        }
        body.extend_from_slice(&buffer[..count]);
    }
    Ok(HttpRequest { method, path, body })
}

fn integer(object: &Map<String, Value>, key: &str, default: usize) -> Result<usize> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v <= i32::MAX as usize)
            .ok_or_else(|| {
                Error::new(format!(
                    "{key} requires a nonnegative integer <= 2147483647"
                ))
            }),
    }
}

fn number(object: &Map<String, Value>, key: &str, default: f32) -> Result<f32> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_f64()
            .map(|v| v as f32)
            .filter(|v| v.is_finite())
            .ok_or_else(|| Error::new(format!("{key} requires a finite number"))),
    }
}

fn parse_input(body: &[u8], chat: bool, options: &Options) -> Result<Input> {
    let value: Value =
        serde_json::from_slice(body).map_err(|e| Error::new(format!("invalid JSON: {e}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| Error::new("JSON body must be an object"))?;
    for key in object.keys() {
        let allowed = matches!(
            key.as_str(),
            "model" | "max_tokens" | "temperature" | "top_p" | "top_k" | "seed" | "stream"
        ) || key == if chat { "messages" } else { "prompt" };
        if !allowed {
            return Err(Error::new(format!("unsupported field: {key}")));
        }
    }
    if let Some(model) = object.get("model") {
        if model.as_str().is_none_or(str::is_empty) {
            return Err(Error::new("model must be a nonempty string"));
        }
        if model.as_str() != Some(model_name(options)) {
            return Err(Error::new("model does not match the loaded GGUF basename"));
        }
    }
    let prompt = if chat {
        let messages = object
            .get("messages")
            .and_then(Value::as_array)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| Error::new("messages must be a nonempty array"))?;
        let mut parsed = Vec::with_capacity(messages.len());
        for message in messages {
            let object = message
                .as_object()
                .ok_or_else(|| Error::new("each message must be an object"))?;
            if object.keys().any(|key| key != "role" && key != "content") {
                return Err(Error::new("messages support only role and content"));
            }
            let role = object
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::new("message role must be a string"))?;
            if !matches!(role, "system" | "user" | "assistant") {
                return Err(Error::new(
                    "message role must be system, user, or assistant",
                ));
            }
            let content = object
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::new("message content must be a string"))?;
            if role.contains('\0') || content.contains('\0') {
                return Err(Error::new("chat messages must not contain NUL"));
            }
            parsed.push((role.to_owned(), content.to_owned()));
        }
        Prompt::Chat(parsed)
    } else {
        let prompt = object
            .get("prompt")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| Error::new("prompt must be a nonempty string"))?;
        Prompt::Text(prompt.to_owned())
    };
    let sampling = SamplingParams {
        temperature: number(object, "temperature", options.sampling.temperature)?,
        top_p: number(object, "top_p", options.sampling.top_p)?,
        top_k: integer(object, "top_k", options.sampling.top_k)?,
        seed: match object.get("seed") {
            None => options.sampling.seed,
            Some(value) => value
                .as_u64()
                .ok_or_else(|| Error::new("seed must be a nonnegative 64-bit integer"))?,
        },
    };
    validate_sampling(&sampling)?;
    let stream = match object.get("stream") {
        None => false,
        Some(value) => value
            .as_bool()
            .ok_or_else(|| Error::new("stream must be a boolean"))?,
    };
    Ok(Input {
        prompt,
        max_tokens: integer(object, "max_tokens", options.max_tokens)?,
        sampling,
        stream,
    })
}

fn error_response(
    socket: &mut TcpStream,
    status: u16,
    message: &str,
    kind: &str,
) -> io::Result<()> {
    json_response(
        socket,
        status,
        &json!({"error":{"message":message,"type":kind}}),
    )
}

fn json_response(socket: &mut TcpStream, status: u16, value: &Value) -> io::Result<()> {
    let body = value.to_string();
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Payload Too Large",
        417 => "Expectation Failed",
        431 => "Request Header Fields Too Large",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    write!(socket, "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
}

fn sse(socket: &mut TcpStream, value: &Value) -> io::Result<()> {
    write!(socket, "data: {value}\n\n")?;
    socket.flush()
}

fn disconnected(socket: &TcpStream) -> bool {
    let _ = socket.set_read_timeout(Some(Duration::from_millis(1)));
    let mut byte = [0u8; 1];
    match socket.peek(&mut byte) {
        Ok(0) => true,
        Ok(_) => false,
        Err(error) => !matches!(
            error.kind(),
            io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::Interrupted
        ),
    }
}

fn handle(
    socket: &mut TcpStream,
    commands: &SyncSender<Command>,
    ids: &AtomicU64,
    options: &Options,
    healthy: &AtomicBool,
    snapshot: &Mutex<Stats>,
) -> io::Result<()> {
    let request = match read_request(socket) {
        Ok(request) => request,
        Err((status, message)) => {
            return error_response(socket, status, &message, "invalid_request_error")
        }
    };
    if request.path == "/health" {
        if request.method != "GET" {
            return error_response(socket, 405, "health requires GET", "invalid_request_error");
        }
        let stats = snapshot
            .lock()
            .map(|value| health_stats(&value))
            .unwrap_or(Value::Null);
        return if healthy.load(Ordering::Acquire) {
            json_response(
                socket,
                200,
                &json!({"status":"ok","model":model_name(options),"stats":stats}),
            )
        } else {
            error_response(socket, 503, "model owner is unavailable", "server_error")
        };
    }
    let chat = request.path == "/v1/chat/completions";
    if !chat && request.path != "/v1/completions" {
        return error_response(socket, 404, "unknown endpoint", "invalid_request_error");
    }
    if request.method != "POST" {
        return error_response(
            socket,
            405,
            "completion endpoints require POST",
            "invalid_request_error",
        );
    }
    let input = match parse_input(&request.body, chat, options) {
        Ok(input) => input,
        Err(error) => {
            return error_response(socket, 400, &error.to_string(), "invalid_request_error")
        }
    };
    let streaming = input.stream;
    let id = ids.fetch_add(1, Ordering::Relaxed);
    let request_id = format!("{}-{id}", if chat { "chatcmpl" } else { "cmpl" });
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (accepted_tx, accepted_rx) = mpsc::sync_channel(1);
    let (events_tx, events_rx) = mpsc::sync_channel(EVENT_QUEUE);
    let cancelled = Arc::new(AtomicBool::new(false));
    let _cancel = CancelOnDrop(Arc::clone(&cancelled));
    match commands.try_send(Command {
        id,
        input,
        accepted: accepted_tx,
        events: events_tx,
        cancelled,
    }) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => {
            return error_response(socket, 503, "model command queue is full", "overloaded")
        }
        Err(TrySendError::Disconnected(_)) => {
            return error_response(socket, 503, "model owner is unavailable", "server_error")
        }
    }
    loop {
        match accepted_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(Ok(())) => break,
            Ok(Err(error)) => {
                let (status, kind) = if error.contains("pending") || error.contains("queue") {
                    (503, "overloaded")
                } else if error.contains("poison")
                    || error.contains("backend")
                    || error.contains("unavailable")
                {
                    (503, "server_error")
                } else {
                    (400, "invalid_request_error")
                };
                return error_response(socket, status, &error, kind);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return error_response(socket, 503, "model owner stopped", "server_error")
            }
            Err(mpsc::RecvTimeoutError::Timeout) if disconnected(socket) => return Ok(()),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
    }
    if streaming {
        write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\nX-Accel-Buffering: no\r\n\r\n")?;
        if chat {
            sse(
                socket,
                &chunk(
                    &request_id,
                    created,
                    model_name(options),
                    true,
                    "",
                    None,
                    Some("assistant"),
                ),
            )?;
        }
    }
    let mut output = Utf8Output::default();
    loop {
        match events_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(Event::Token { bytes, .. }) => {
                let text = output.push(&bytes);
                if streaming && !text.is_empty() {
                    sse(
                        socket,
                        &chunk(
                            &request_id,
                            created,
                            model_name(options),
                            chat,
                            &text,
                            None,
                            None,
                        ),
                    )?;
                }
                if disconnected(socket) {
                    return Ok(());
                }
            }
            Ok(Event::Finished {
                reason,
                prompt_tokens,
                generated_tokens,
                cached_tokens,
                prefill_tokens,
                ..
            }) => {
                let tail = output.finish();
                let reason = reason.to_string();
                let usage = usage(
                    prompt_tokens,
                    generated_tokens,
                    cached_tokens,
                    prefill_tokens,
                );
                if streaming {
                    if !tail.is_empty() {
                        sse(
                            socket,
                            &chunk(
                                &request_id,
                                created,
                                model_name(options),
                                chat,
                                &tail,
                                None,
                                None,
                            ),
                        )?;
                    }
                    let mut finished = chunk(
                        &request_id,
                        created,
                        model_name(options),
                        chat,
                        "",
                        Some(&reason),
                        None,
                    );
                    finished["usage"] = usage;
                    sse(socket, &finished)?;
                    socket.write_all(b"data: [DONE]\n\n")?;
                    return socket.flush();
                }
                let choice = if chat {
                    json!({"index":0,"message":{"role":"assistant","content":output.text},"finish_reason":reason})
                } else {
                    json!({"index":0,"text":output.text,"finish_reason":reason})
                };
                return json_response(
                    socket,
                    200,
                    &json!({"id":request_id,"object":if chat {"chat.completion"} else {"text_completion"},
                    "created":created,"model":model_name(options),"choices":[choice],"usage":usage}),
                );
            }
            Ok(Event::Error { message, .. }) => {
                return stream_error(socket, streaming, &message);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return stream_error(
                    socket,
                    streaming,
                    "generation stopped or client output buffer overflowed",
                )
            }
            Err(mpsc::RecvTimeoutError::Timeout) if disconnected(socket) => return Ok(()),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
    }
}

fn stream_error(socket: &mut TcpStream, streaming: bool, message: &str) -> io::Result<()> {
    if streaming {
        sse(
            socket,
            &json!({"error":{"message":message,"type":"server_error"}}),
        )?;
        socket.write_all(b"data: [DONE]\n\n")
    } else {
        error_response(socket, 500, message, "server_error")
    }
}

fn model_name(options: &Options) -> &str {
    std::path::Path::new(&options.model)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(&options.model)
}

fn health_stats(stats: &Stats) -> Value {
    json!({"forward_calls":stats.forward_batches,"active_requests":stats.active,"queued_requests":stats.pending,
        "reserved_tokens":stats.reserved_tokens,"cache_entries":stats.cached_entries,"cache_tokens":stats.cached_tokens,
        "cached_tokens":stats.cache_hit_tokens,"prefill_tokens":stats.prefill_tokens,"decode_tokens":stats.decode_tokens,
        "total_submitted":stats.total_submitted,"total_finished":stats.total_completed,
        "total_cancelled":stats.total_cancelled,"total_failed":stats.total_failed,"free_sequences":stats.free_sequences})
}

fn chunk(
    id: &str,
    created: u64,
    model: &str,
    chat: bool,
    text: &str,
    reason: Option<&str>,
    role: Option<&str>,
) -> Value {
    let choice = if chat {
        let mut delta = Map::new();
        if let Some(role) = role {
            delta.insert("role".into(), json!(role));
        }
        if !text.is_empty() {
            delta.insert("content".into(), json!(text));
        }
        json!({"index":0,"delta":delta,"finish_reason":reason})
    } else {
        json!({"index":0,"text":text,"finish_reason":reason})
    };
    json!({"id":id,"object":if chat {"chat.completion.chunk"} else {"text_completion"},
        "created":created,"model":model,"choices":[choice]})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_http(wire: &[u8]) -> std::result::Result<HttpRequest, (u16, String)> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind test listener");
        let address = listener.local_addr().expect("test listener address");
        let wire = wire.to_vec();
        let sender = thread::spawn(move || {
            let mut client = TcpStream::connect(address).expect("connect test socket");
            client.write_all(&wire).expect("write test request");
            client
                .shutdown(std::net::Shutdown::Write)
                .expect("end test request");
        });
        let (mut socket, _) = listener.accept().expect("accept test socket");
        socket
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("test timeout");
        let parsed = read_request(&mut socket);
        sender.join().expect("request sender finished");
        parsed
    }

    #[test]
    fn http_framing_rejects_ambiguous_and_unbounded_bodies() {
        for (wire, expected) in [
            (&b"POST /v1/completions HTTP/1.1\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}"[..], 400),
            (&b"POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"[..], 400),
            (&b"POST /v1/completions HTTP/1.1\r\nContent-Length: 1048577\r\n\r\n"[..], 413),
            (&b"POST /v1/completions HTTP/1.1\r\n\r\n"[..], 411),
            (&b"POST /v1/completions HTTP/1.1\r\nContent-Length: 5\r\n\r\n{}"[..], 400),
        ] {
            let error = parse_http(wire).err().expect("invalid request rejected");
            assert_eq!(error.0, expected, "{error:?}");
        }
    }

    #[test]
    fn http_body_respects_declared_byte_count() {
        let request =
            parse_http(b"POST /v1/completions HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}ignored")
                .expect("valid request");
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/completions");
        assert_eq!(request.body, b"{}");
    }

    #[test]
    fn json_parameters_are_strict() {
        let options = Options::default();
        for body in [
            r#"{"prompt":"a","max_tokens":-1}"#,
            r#"{"prompt":"a","max_tokens":1.0}"#,
            r#"{"prompt":"a","stream":"true"}"#,
            r#"{"prompt":"a","temperature":-0.5}"#,
            r#"{"prompt":"a","stop":"x"}"#,
            r#"{"prompt":["a"]}"#,
        ] {
            assert!(
                parse_input(body.as_bytes(), false, &options).is_err(),
                "{body}"
            );
        }
        assert!(parse_input(br#"{"prompt":"a","max_tokens":0}"#, false, &options).is_ok());
    }
    #[test]
    fn chat_requires_supported_role_and_text_content() {
        let options = Options::default();
        assert!(parse_input(
            br#"{"messages":[{"role":"user","content":"hello"}]}"#,
            true,
            &options
        )
        .is_ok());
        assert!(parse_input(
            br#"{"messages":[{"role":"tool","content":"hello"}]}"#,
            true,
            &options
        )
        .is_err());
        assert!(parse_input(
            br#"{"messages":[{"role":"user","content":[]}] }"#,
            true,
            &options
        )
        .is_err());
    }
}
