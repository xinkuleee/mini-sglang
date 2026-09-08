#ifndef MINISGL_BRIDGE_H
#define MINISGL_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(MSGL_SHARED)
#ifdef MSGL_BUILD
#define MSGL_API __declspec(dllexport)
#else
#define MSGL_API __declspec(dllimport)
#endif
#else
#define MSGL_API
#endif
#ifdef __cplusplus
#define MSGL_NOEXCEPT noexcept
extern "C" {
#else
#define MSGL_NOEXCEPT
#endif

/* 薄计算边界：只暴露模型、词表、logits 和 KV 引用；不包含调度和采样。
 * 所有函数捕获 C++ 异常。错误为 -1；0 成功；1 表示输出缓冲区不足。
 * error() 返回线程局部错误，下次同线程调用前有效。handle 由单一线程独占。 */
typedef struct msgl_handle msgl_handle;
typedef struct msgl_config {
    const char *model_path;
    uint32_t context_tokens;
    uint32_t batch_size;
    uint32_t max_sequences;
    int32_t threads;
    int32_t gpu_layers;
} msgl_config;
typedef struct msgl_batch_token {
    int32_t token;
    int32_t position;
    int32_t sequence;
    int32_t logits;
} msgl_batch_token;
typedef struct msgl_chat_message {
    const char *role;
    const char *content;
} msgl_chat_message;

MSGL_API const char *msgl_error(void) MSGL_NOEXCEPT;
MSGL_API msgl_handle *msgl_create(const msgl_config *config) MSGL_NOEXCEPT;
MSGL_API void msgl_destroy(msgl_handle *handle) MSGL_NOEXCEPT;
MSGL_API int32_t msgl_vocab_size(msgl_handle *handle) MSGL_NOEXCEPT;
MSGL_API int32_t msgl_tokenize(msgl_handle *, const char *text, size_t text_len, int32_t *output,
                               size_t capacity, size_t *required) MSGL_NOEXCEPT;
/* piece 是原始字节，不保证每个 token 单独构成 UTF-8；不写 NUL 结尾。 */
MSGL_API int32_t msgl_piece(msgl_handle *, int32_t token, char *output, size_t capacity,
                            size_t *required) MSGL_NOEXCEPT;
MSGL_API int32_t msgl_is_eog(msgl_handle *, int32_t token) MSGL_NOEXCEPT;
/* output 按输入顺序，只包含 logits!=0 的行，行宽为 vocab_size。 */
MSGL_API int32_t msgl_forward(msgl_handle *, const msgl_batch_token *, size_t count, float *output,
                              size_t output_count) MSGL_NOEXCEPT;
/* destination 必须为空，source 由 scheduler 保证从 position=0 连续写入。 */
MSGL_API int32_t msgl_copy_seq(msgl_handle *, int32_t source, int32_t destination,
                               int32_t end_exclusive) MSGL_NOEXCEPT;
MSGL_API int32_t msgl_remove_seq(msgl_handle *, int32_t sequence) MSGL_NOEXCEPT;
MSGL_API int32_t msgl_apply_chat_template(msgl_handle *, const msgl_chat_message *, size_t count,
                                          int32_t add_generation_prompt, char *output,
                                          size_t capacity, size_t *required) MSGL_NOEXCEPT;

#ifdef __cplusplus
}
#endif
#endif
