#pragma once

// 单线程拥有者推进一个确定性的状态机：先 decode，再分块 prefill。
// 每次 step 至多一次前向；完整输出预算提前预留，避免生成中途无 KV 可用。
// HTTP 等并发入口应通过队列把操作交给同一个拥有者，不能并发调用本类。
#include "minisgl/engine.hpp"
#include <memory>
#include <vector>

namespace minisgl {
class Scheduler {
  public:
    explicit Scheduler(Engine &engine, SchedulerConfig config = {});
    ~Scheduler();
    Scheduler(const Scheduler &) = delete;
    Scheduler &operator=(const Scheduler &) = delete;
    Scheduler(Scheduler &&) = delete;
    Scheduler &operator=(Scheduler &&) = delete;

    // 验证/分词后入有界队列；非法输入、重复 ID、队列满抛异常且不入队。
    RequestId submit(GenerationRequest request);
    // 返回是否找到未完成的请求。取消完成事件在下次 step 交付。
    bool cancel(const RequestId &id);
    std::vector<Event> step();
    bool idle() const;
    SchedulerStats stats() const;
    // 用于教学调试和语义测试；不变量破坏时抛 logic_error。
    void assert_invariants() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace minisgl
