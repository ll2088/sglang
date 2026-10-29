"""Helpers for streaming token histories without per-step full copies."""

from __future__ import annotations

from typing import Any, Sequence


def token_history_view(token_ids: Sequence[Any] | None, sent_count: int):
    """Read-only view of the not-yet-sent suffix of a token history.

    Streaming paths used to copy the complete token history on every update
    (``list(token_ids)``), which is O(n) per step. Callers only ever consume
    the suffix after ``sent_count``, so return a slice of the existing
    sequence instead of a fresh full copy. The returned object aliases the
    source buffer: callers must not mutate it, and must not retain it past
    the next update of ``token_ids``.
    """
    if token_ids is None:
        return ()
    if sent_count <= 0:
        return token_ids
    return token_ids[sent_count:]


def token_history_len(token_ids: Sequence[Any] | None) -> int:
    """Length of a token history that may be ``None`` or empty."""
    return 0 if token_ids is None else len(token_ids)
