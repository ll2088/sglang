# Streaming performance notes

This document records three low-risk optimizations borrowed from upstream
gRPC work. They are additive: no public API, protocol, or streaming-interval
change is included.

## Native protobuf

SMG and gRPC launchers warn once at startup when the pure-Python protobuf
implementation is active, then continue startup. Detection is optional: if the
private `google.protobuf.internal.api_implementation` API cannot be imported or
has no `Type` attribute, the warning is skipped.

Set the implementation before starting Python:

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb
```

Changing the variable after protobuf has been imported does not change the
active implementation. Native `upb` and `cpp` do not produce a warning. The
installed protobuf wheel must include the native extension.

## Zero-copy streaming token history

Streaming updates must not copy the full token history on every step. Use the
helpers in `sglang.srt.utils.streaming`:

- `token_history_view(token_ids)` returns a read-only view (aliases the
  engine-owned buffer; `None`/empty map to `()`).
- `new_token_slice(token_ids, sent_count)` returns only the new suffix.
- `aligned_logprob_slice(logprobs, sent_count)` keeps logprob alignment.

Token order, text deltas, logprob alignment, terminal metadata, cancellation,
and slow-consumer isolation are preserved.

## Queue notifications via `call_soon_threadsafe`

Shared result queues schedule a direct event-loop callback instead of a
coroutine per notification. This keeps queue order, existing batch
deduplication, event-loop ownership, and shutdown handling. A closed loop
raises the existing shutdown error rather than silently dropping the
notification.
