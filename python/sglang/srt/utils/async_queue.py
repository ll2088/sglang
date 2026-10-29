"""Cross-thread notification helpers for asyncio queues.

Notifications are scheduled with ``loop.call_soon_threadsafe`` instead of
``asyncio.run_coroutine_threadsafe`` so a single notification does not pay
for coroutine scheduling. Ordering, batch deduplication, event-loop ownership
and shutdown handling are preserved.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, List


class EventLoopShutdownError(RuntimeError):
    """Raised when a notification targets a stopped or closed event loop."""


def _notify_many(queues: Iterable[asyncio.Queue]) -> None:
    for queue in queues:
        queue.put_nowait(None)


def notify_one(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
    """Schedule a single queue notification on ``loop`` from another thread."""
    if not loop.is_running():
        raise EventLoopShutdownError()
    try:
        loop.call_soon_threadsafe(queue.put_nowait, None)
    except RuntimeError as error:
        if loop.is_closed():
            raise EventLoopShutdownError() from error
        raise


def notify_many(
    loop: asyncio.AbstractEventLoop, queues: List[asyncio.Queue]
) -> None:
    """Schedule one batched notification for ``queues`` on ``loop``.

    ``frozenset`` deduplicates repeated queue references while keeping the
    callback a single event-loop turn.
    """
    if not loop.is_running():
        raise EventLoopShutdownError()
    try:
        loop.call_soon_threadsafe(_notify_many, frozenset(queues))
    except RuntimeError as error:
        if loop.is_closed():
            raise EventLoopShutdownError() from error
        raise
