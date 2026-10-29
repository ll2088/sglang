"""Thread-safe asyncio queue notification helpers.

Shared result queues used to schedule a coroutine per notification via
``asyncio.run_coroutine_threadsafe``. For a single notification that costs a
full coroutine round-trip; scheduling the callback directly with
``call_soon_threadsafe`` is cheaper while preserving ordering, batch
deduplication, event-loop ownership and shutdown handling.

This module is intentionally dependency-free (standard library only) so it can
be unit-tested without importing the rest of sglang.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, List


class EventLoopShutdownError(RuntimeError):
    """Raised when a notification targets a loop that is not running."""


class _Notifier:
    """Owns one asyncio.Event and the loop it belongs to."""

    __slots__ = ("_event", "_loop")

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._event = asyncio.Event()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def notify(self) -> None:
        """Set the event. Must run on the owning loop's thread."""
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()
        self._event.clear()


def _notify_many(notifiers: Iterable[_Notifier]) -> None:
    """Set every notifier's event in one callback (batch dedup by caller)."""
    for notifier in notifiers:
        notifier.notify()


def notify_one(loop: asyncio.AbstractEventLoop, notifier: _Notifier) -> None:
    """Schedule a single notification on ``loop`` from any thread."""
    if not loop.is_running():
        raise EventLoopShutdownError()
    try:
        loop.call_soon_threadsafe(notifier.notify)
    except RuntimeError as error:
        if loop.is_closed():
            raise EventLoopShutdownError() from error
        raise


def notify_many(
    loop: asyncio.AbstractEventLoop, notifiers: List[_Notifier]
) -> None:
    """Schedule one batched notification for ``notifiers`` on ``loop``.

    ``frozenset`` keeps the existing batch deduplication; ordering of the
    remaining notifiers is not observable because each only sets its own event.
    """
    if not loop.is_running():
        raise EventLoopShutdownError()
    try:
        loop.call_soon_threadsafe(_notify_many, frozenset(notifiers))
    except RuntimeError as error:
        if loop.is_closed():
            raise EventLoopShutdownError() from error
        raise
