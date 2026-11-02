"""Tests that queue notifications use call_soon_threadsafe, not coroutines."""

from __future__ import annotations

import asyncio

from sglang.srt.utils import async_queue


def test_notify_one_uses_call_soon_threadsafe(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeLoop:
        def is_running(self) -> bool:
            return True

        def is_closed(self) -> bool:
            return False

        def call_soon_threadsafe(self, callback, *args) -> None:
            calls.append((callback, args))

    queue: asyncio.Queue = asyncio.Queue()
    async_queue.notify_one(FakeLoop(), queue)  # type: ignore[arg-type]
    assert len(calls) == 1
    callback, args = calls[0]
    assert callback == queue.put_nowait
    assert args == (None,)


def test_notify_many_uses_single_callback(monkeypatch) -> None:
    calls: list[tuple] = []

    class FakeLoop:
        def is_running(self) -> bool:
            return True

        def is_closed(self) -> bool:
            return False

        def call_soon_threadsafe(self, callback, *args) -> None:
            calls.append((callback, args))

    queues: list[asyncio.Queue] = [asyncio.Queue(), asyncio.Queue()]
    async_queue.notify_many(FakeLoop(), queues + queues)  # type: ignore[arg-type]
    assert len(calls) == 1
    callback, args = calls[0]
    assert callback is async_queue._notify_many
    assert isinstance(args[0], frozenset)
    assert len(args[0]) == 2


def test_closed_loop_during_schedule_raises_shutdown_error(monkeypatch) -> None:
    class FakeLoop:
        def is_running(self) -> bool:
            return True

        def is_closed(self) -> bool:
            return True

        def call_soon_threadsafe(self, callback, *args) -> None:
            raise RuntimeError("Event loop is closed")

    queue: asyncio.Queue = asyncio.Queue()
    try:
        async_queue.notify_one(FakeLoop(), queue)  # type: ignore[arg-type]
    except async_queue.EventLoopShutdownError:
        pass
    else:
        raise AssertionError("expected EventLoopShutdownError")
