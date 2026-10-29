"""Helpers for streaming token delivery.

Streaming updates must not copy the full token history on every step.  These
helpers expose read-only views over the engine-owned history so callers can
serialize only the new suffix while keeping token order, logprob alignment
and terminal metadata intact.
"""

from typing import Any, Optional, Sequence


def token_history_view(token_ids: Optional[Sequence[int]]) -> Sequence[int]:
    """Return a read-only view of ``token_ids`` without copying.

    ``None`` and empty histories map to an empty tuple.  The returned object
    must not be mutated by callers; it aliases the engine-owned buffer.
    """
    if not token_ids:
        return ()
    return token_ids


def new_token_slice(
    token_ids: Optional[Sequence[int]], sent_count: int
) -> Sequence[int]:
    """Return the tokens produced since ``sent_count`` without copying.

    Slicing a list copies only the new suffix, never the full history.
    """
    if not token_ids:
        return ()
    if sent_count <= 0:
        return token_ids
    if sent_count >= len(token_ids):
        return ()
    return token_ids[sent_count:]


def aligned_logprob_slice(
    logprobs: Optional[Sequence[Any]], sent_count: int
) -> Sequence[Any]:
    """Return the logprob entries aligned with :func:`new_token_slice`."""
    if not logprobs:
        return ()
    if sent_count <= 0:
        return logprobs
    if sent_count >= len(logprobs):
        return ()
    return logprobs[sent_count:]
