# gRPC 性能优化说明

本仓库在 gRPC/流式服务路径上借鉴了上游三项优化，均保持现有协议与公开 API 不变。

## 原生 protobuf

推荐在启动 Python 前设置环境变量：

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb
```

- 启动时检测当前 protobuf 实现；若为纯 Python 实现则输出一次告警并继续启动。
- 原生 `upb` 与 `cpp` 不产生告警。
- 若私有检测 API 不可用（导入失败或无 `Type` 属性），跳过告警并继续启动。
- 环境变量必须在导入 protobuf 之前生效；导入后修改不会切换实现。
- 所用 protobuf wheel 需包含原生扩展。

## 流式 token 历史零拷贝

流式响应构造时使用只读视图与后缀切片，避免每轮复制完整 token 历史：

- `token_history_view` 返回引擎持有缓冲区的只读视图。
- `new_token_slice` / `aligned_logprob_slice` 仅复制新增后缀。
- token 顺序、logprob 对齐、终止元数据与取消行为保持不变。

## 队列通知使用 call_soon_threadsafe

跨线程通知共享结果队列时，直接调度事件循环回调，而非为每次通知创建协程：

- 单次通知使用 `loop.call_soon_threadsafe(queue.notify)`。
- 批量通知使用 `loop.call_soon_threadsafe(_notify_many, frozenset(queues))`，保留去重。
- 保留队列顺序、事件循环归属与关闭处理；事件循环已关闭时抛出 `EventLoopShutdownError`。
