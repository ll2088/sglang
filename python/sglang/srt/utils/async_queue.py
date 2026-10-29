"""Cross-thread notification helpers for asyncio queues.

Scheduling a coroutine per notification (``run_coroutine_threadsafe``) adds a
context switch per item.  Scheduling the queue's ``notify`` callback directly
with ``call_soon_threadsafe`` keeps ordering, batch deduplication, event-loop
ownership and shutdown handling while cutting that overhead.
"""

import asyncio
from typing import Iterable, List


class EventLoopShutdownError(RuntimeError):
    """Raised when a notification targets a stopped or closed event loop."""


def _notify_many(queues: Iterable) -> None:
    for queue in queues:
        queue.notify()


def notify_one(loop: asyncio.AbstractEventLoop, queue) -> None:
    """Schedule a single queue notification on ``loop`` from another thread."""
    if not loop.is_running():
        raise EventLoopShutdownError()
    try:
        loop.call_soon_threadsafe(queue.notify)
    except RuntimeError as error:
        if loop.is_closed():
            raise EventLoopShutdownError() from error
        raise


def notify_many(loop: asyncio.AbstractEventLoop, queues: List) -> None:
    """Schedule one callback that notifies every queue, deduplicated."""
    if not loop.is_running():
        raise EventLoopShutdownError()
    try:
        loop.call_soon_threadsafe(_notify_many, frozenset(queues))
    except RuntimeError as error:
        if loop.is_closed():
            raise EventLoopShutdownError() from error
        raise
