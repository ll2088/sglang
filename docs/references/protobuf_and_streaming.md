# Protobuf, zero-copy streaming and queue notifications

This page documents three performance-related behaviors in SGLang. They are
internal optimizations: no public API or wire protocol changes are involved.

## Native protobuf for gRPC

Native protobuf (`upb`, or `cpp`) is recommended for gRPC and streaming
throughput. Set the environment variable **before starting Python**:

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb
```

The installed protobuf wheel must include the native extension. Changing the
variable after protobuf has been imported does not change the active
implementation. When the pure Python implementation is active, the OpenAI
gRPC entrypoint logs a single warning and continues startup. Detection is
best-effort: if the private protobuf implementation API is unavailable, the
warning is skipped and startup proceeds. `upb` and `cpp` do not warn.

## Zero-copy streaming token history

Streaming responses must not copy the full token history on every update.
`python/sglang/srt/utils/streaming.py` exposes read-only views and suffix
slices so only the newly produced tokens are serialized. Token order, text
deltas, logprob alignment, terminal metadata, cancellation and slow-consumer
isolation are preserved.

## Queue notifications with `call_soon_threadsafe`

Cross-thread queue notifications schedule the queue's `notify` callback
directly on the event loop with `call_soon_threadsafe` instead of scheduling a
coroutine per notification. Ordering, batch deduplication, event-loop
ownership and shutdown handling are preserved; a stopped or closed loop raises
`EventLoopShutdownError`.
