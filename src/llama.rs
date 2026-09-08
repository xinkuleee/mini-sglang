//! 唯一 FFI 模块：拥有一个计算 context，将 C ABI 结果转换为 Rust 的 Result。
//! tokenizer/logits/KV 引用由 llama.cpp 提供；采样、radix、调度均在 Rust 中。
use crate::engine::{BatchToken, Engine};
use crate::types::{Error, Result};
use std::cell::Cell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::marker::PhantomData;
use std::ptr::NonNull;

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub model_path: String,
    pub context_tokens: u32,
    pub batch_size: u32,
    pub max_sequences: u32,
    pub threads: i32,
    pub gpu_layers: i32,
}
impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            context_tokens: 4096,
            batch_size: 512,
            max_sequences: 64,
            threads: 4,
            gpu_layers: 0,
        }
    }
}
#[repr(C)]
struct NativeConfig {
    model_path: *const c_char,
    context_tokens: u32,
    batch_size: u32,
    max_sequences: u32,
    threads: i32,
    gpu_layers: i32,
}
#[repr(C)]
struct NativeToken {
    token: i32,
    position: i32,
    sequence: i32,
    logits: i32,
}
#[repr(C)]
struct NativeChatMessage {
    role: *const c_char,
    content: *const c_char,
}

unsafe extern "C" {
    fn msgl_error() -> *const c_char;
    fn msgl_create(config: *const NativeConfig) -> *mut c_void;
    fn msgl_destroy(handle: *mut c_void);
    fn msgl_vocab_size(handle: *mut c_void) -> i32;
    fn msgl_tokenize(
        handle: *mut c_void,
        text: *const c_char,
        text_len: usize,
        output: *mut i32,
        capacity: usize,
        required: *mut usize,
    ) -> i32;
    fn msgl_piece(
        handle: *mut c_void,
        token: i32,
        output: *mut c_char,
        capacity: usize,
        required: *mut usize,
    ) -> i32;
    fn msgl_is_eog(handle: *mut c_void, token: i32) -> i32;
    fn msgl_forward(
        handle: *mut c_void,
        batch: *const NativeToken,
        count: usize,
        output: *mut f32,
        output_count: usize,
    ) -> i32;
    fn msgl_copy_seq(handle: *mut c_void, src: i32, dst: i32, end: i32) -> i32;
    fn msgl_remove_seq(handle: *mut c_void, seq: i32) -> i32;
    fn msgl_apply_chat_template(
        handle: *mut c_void,
        messages: *const NativeChatMessage,
        count: usize,
        add_generation_prompt: i32,
        output: *mut c_char,
        capacity: usize,
        required: *mut usize,
    ) -> i32;
}
fn native_error() -> Error {
    // SAFETY: msgl_error returns a NUL-terminated, thread-local buffer valid until
    // the next native call. Copy it immediately so Result owns the error text.
    unsafe { Error::new(CStr::from_ptr(msgl_error()).to_string_lossy().into_owned()) }
}
fn checked(status: i32) -> Result<i32> {
    if status < 0 {
        Err(native_error())
    } else {
        Ok(status)
    }
}
fn exact(status: i32) -> Result<()> {
    if checked(status)? == 0 {
        Ok(())
    } else {
        Err(Error::new("native output size changed unexpectedly"))
    }
}
fn i32_value(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::new("value exceeds llama int32 range"))
}
fn c_string(value: &str) -> Result<CString> {
    CString::new(value).map_err(|_| Error::new("model path and chat fields cannot contain NUL"))
}

pub struct LlamaEngine {
    handle: NonNull<c_void>,
    vocab_size: usize,
    // !Sync: one scheduler owns a mutable KV cache. Ownership may move threads.
    _not_sync: PhantomData<Cell<()>>,
}
// SAFETY: the context has a unique owner; no native pointer escapes this module,
// and all KV mutation requires &mut self. llama contexts are movable across threads.
unsafe impl Send for LlamaEngine {}
impl LlamaEngine {
    pub fn new(config: LlamaConfig) -> Result<Self> {
        let path = c_string(&config.model_path)?;
        let native = NativeConfig {
            model_path: path.as_ptr(),
            context_tokens: config.context_tokens,
            batch_size: config.batch_size,
            max_sequences: config.max_sequences,
            threads: config.threads,
            gpu_layers: config.gpu_layers,
        };
        // SAFETY: native borrows path only during create; returned handle is owned.
        let handle = NonNull::new(unsafe { msgl_create(&native) }).ok_or_else(native_error)?;
        let mut engine = Self {
            handle,
            vocab_size: 0,
            _not_sync: PhantomData,
        };
        // SAFETY: engine owns a live handle. Drop frees it if validation fails.
        let count = checked(unsafe { msgl_vocab_size(engine.handle.as_ptr()) })?;
        if count == 0 {
            return Err(Error::new("model vocabulary is empty"));
        }
        engine.vocab_size = count as usize;
        Ok(engine)
    }
    pub fn apply_chat_template(
        &self,
        messages: &[(&str, &str)],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let strings: Vec<(CString, CString)> = messages
            .iter()
            .map(|(role, content)| Ok((c_string(role)?, c_string(content)?)))
            .collect::<Result<_>>()?;
        let native: Vec<NativeChatMessage> = strings
            .iter()
            .map(|(role, content)| NativeChatMessage {
                role: role.as_ptr(),
                content: content.as_ptr(),
            })
            .collect();
        let mut required = 0;
        // SAFETY: all role/content CStrings outlive both native calls; the first
        // call reports buffer length and the second writes within that capacity.
        checked(unsafe {
            msgl_apply_chat_template(
                self.handle.as_ptr(),
                native.as_ptr(),
                native.len(),
                i32::from(add_generation_prompt),
                std::ptr::null_mut(),
                0,
                &mut required,
            )
        })?;
        let mut output = vec![0_u8; required];
        exact(unsafe {
            msgl_apply_chat_template(
                self.handle.as_ptr(),
                native.as_ptr(),
                native.len(),
                i32::from(add_generation_prompt),
                output.as_mut_ptr().cast(),
                output.len(),
                &mut required,
            )
        })?;
        output.truncate(required);
        String::from_utf8(output)
            .map_err(|_| Error::new("model chat template produced invalid UTF-8"))
    }
}
impl Drop for LlamaEngine {
    fn drop(&mut self) {
        // SAFETY: no Clone/Copy exists; this is the unique final release.
        unsafe { msgl_destroy(self.handle.as_ptr()) };
    }
}
impl Engine for LlamaEngine {
    fn tokenize(&mut self, text: &str) -> Result<Vec<i32>> {
        let mut required = 0;
        // SAFETY: text is borrowed with an explicit byte length (embedded NUL is
        // valid); output allocation follows the size reported by the same vocab.
        checked(unsafe {
            msgl_tokenize(
                self.handle.as_ptr(),
                text.as_ptr().cast(),
                text.len(),
                std::ptr::null_mut(),
                0,
                &mut required,
            )
        })?;
        let mut output = vec![0; required];
        exact(unsafe {
            msgl_tokenize(
                self.handle.as_ptr(),
                text.as_ptr().cast(),
                text.len(),
                output.as_mut_ptr(),
                output.len(),
                &mut required,
            )
        })?;
        output.truncate(required);
        Ok(output)
    }
    fn piece(&self, token: i32) -> Result<Vec<u8>> {
        let mut required = 0;
        // SAFETY: the size/write protocol is bounded by output.len(); raw bytes
        // are preserved because one token may end in the middle of UTF-8.
        checked(unsafe {
            msgl_piece(
                self.handle.as_ptr(),
                token,
                std::ptr::null_mut(),
                0,
                &mut required,
            )
        })?;
        let mut output = vec![0; required];
        exact(unsafe {
            msgl_piece(
                self.handle.as_ptr(),
                token,
                output.as_mut_ptr().cast(),
                output.len(),
                &mut required,
            )
        })?;
        output.truncate(required);
        Ok(output)
    }
    fn is_eog(&self, token: i32) -> bool {
        // The trait is infallible. Invalid IDs are never EOG; forward/piece still
        // validate them and return errors rather than indexing native memory.
        unsafe { msgl_is_eog(self.handle.as_ptr(), token) == 1 }
    }
    fn forward(&mut self, batch: &[BatchToken]) -> Result<Vec<Vec<f32>>> {
        let native: Vec<NativeToken> = batch
            .iter()
            .map(|item| {
                Ok(NativeToken {
                    token: item.token,
                    position: i32_value(item.position)?,
                    sequence: i32_value(item.sequence as usize)?,
                    logits: i32::from(item.logits),
                })
            })
            .collect::<Result<_>>()?;
        let rows = batch.iter().filter(|item| item.logits).count();
        let length = rows
            .checked_mul(self.vocab_size)
            .ok_or_else(|| Error::new("logits buffer size overflow"))?;
        let mut output = vec![0.0; length];
        // SAFETY: input and output vectors are stable for the call. The bridge
        // checks batch, token/position/sequence IDs, and exact logits capacity.
        exact(unsafe {
            msgl_forward(
                self.handle.as_ptr(),
                native.as_ptr(),
                native.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        })?;
        Ok(output
            .chunks_exact(self.vocab_size)
            .map(<[f32]>::to_vec)
            .collect())
    }
    fn copy_sequence(&mut self, src: u32, dst: u32, end_exclusive: usize) -> Result<()> {
        // SAFETY: IDs are range checked by the bridge; KV cells remain native.
        exact(unsafe {
            msgl_copy_seq(
                self.handle.as_ptr(),
                i32_value(src as usize)?,
                i32_value(dst as usize)?,
                i32_value(end_exclusive)?,
            )
        })
    }
    fn remove_sequence(&mut self, id: u32) -> Result<()> {
        // SAFETY: only removes this sequence's references, not another owner's KV.
        exact(unsafe { msgl_remove_seq(self.handle.as_ptr(), i32_value(id as usize)?) })
    }
}
