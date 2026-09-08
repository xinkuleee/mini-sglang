#include "minisgl/types.hpp"
#include <limits>
#include <stdexcept>

namespace minisgl {
void validate_config(const SchedulerConfig &c) {
    if (!c.max_running_requests || !c.max_pending_requests || !c.max_batch_tokens ||
        !c.prefill_chunk_size || !c.max_sequence_tokens || !c.max_total_tokens || !c.max_sequences)
        throw std::invalid_argument("scheduler capacities must be positive");
    if (c.max_running_requests > c.max_batch_tokens || c.max_running_requests > c.max_sequences)
        throw std::invalid_argument(
            "running request limit must fit both batch and sequence capacity");
    if (c.max_sequences > static_cast<std::size_t>(std::numeric_limits<SequenceId>::max()))
        throw std::invalid_argument("sequence capacity exceeds backend ID range");
    if (c.max_sequence_tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::invalid_argument("sequence context exceeds backend position range");
}
} // namespace minisgl
