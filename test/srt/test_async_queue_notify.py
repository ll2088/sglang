"""Tests for cross-thread asyncio queue notifications."""

from __future__ import annotations

import asyncio

import pytest

from sglang.srt.utils.async_queue import (
    EventLoopShutdownError,
    notify_many,
    notify_one,
)


def test_notify_one_delivers_in_order() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def produce() -> None:
            for value in range(10):
                queue.put_nowait(value)
                notify_one(loop, queue)

        await asyncio.to_thread(produce)
        values = [await asyncio.wait_for(queue.get(), 5) for _ in range(10)]
        assert values == list(range(10))

    asyncio.run(run())


def test_notify_many_deduplicates_and_delivers() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        queues: list[asyncio.Queue] = [asyncio.Queue(), asyncio.Queue()]

        def produce() -> None:
            for queue in queues:
                queue.put_nowait("x")
            notify_many(loop, queues + queues)

        await asyncio.to_thread(produce)
        for queue in queues:
            assert await asyncio.wait_for(queue.get(), 5) == "x"

    asyncio.run(run())


def test_notify_on_stopped_loop_raises() -> None:
    loop = asyncio.new_event_loop()
    try:
        queue: asyncio.Queue = asyncio.Queue()
        with pytest.raises(EventLoopShutdownError):
            notify_one(loop, queue)
        with pytest.raises(EventLoopShutdownError):
            notify_many(loop, [queue])
    finally:
        loop.close()
