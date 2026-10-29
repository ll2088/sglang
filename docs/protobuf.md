# gRPC 性能优化

SMG/gRPC 路径包含三项优化。

## 原生 protobuf

启动前设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb`，并确保安装的 protobuf wheel 包含原生扩展。
纯 Python 实现会在启动时告警一次但不阻断启动；原生 `upb` 与 `cpp` 不告警。
检测是可选的：若私有实现 API 无法导入或没有 `Type` 属性，则跳过告警并继续启动。
在导入 protobuf 之后再修改该环境变量不会改变已生效的实现。

## 流式 token 历史零拷贝

流式响应只消费已发送位置之后的后缀，使用只读视图与后缀切片（`token_history_view`），
避免每轮对完整 token 历史做全量复制。返回对象与源缓冲区共享内存，调用方不得修改，
也不得在源缓冲区下一次更新后继续持有。token 顺序、logprob 对齐与终止元数据保持不变。

## 队列通知使用 call_soon_threadsafe

共享结果队列的单次通知改为 `loop.call_soon_threadsafe(notifier.notify)`，批量通知合并为一次回调
（`frozenset` 去重）。保留队列顺序、事件循环归属与关闭处理：事件循环未运行或已关闭时抛出
`EventLoopShutdownError`。
