"""Unit tests for call_soon_threadsafe based queue notifications."""

import asyncio

import pytest

from sglang.srt.utils.async_queue import (
    EventLoopShutdownError,
    notify_many,
    notify_one,
)


def test_notify_one_cross_thread_ordering() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def produce() -> None:
            for value in range(10):
                queue.put_nowait(value)
                notify_one(loop, queue)

        await asyncio.to_thread(produce)
        received = [await asyncio.wait_for(queue.get(), 5) for _ in range(10)]
        assert received == list(range(10))

    asyncio.run(run())


def test_notify_many_deduplicates_and_preserves_order() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        queues = [asyncio.Queue(), asyncio.Queue()]

        def produce() -> None:
            for value in range(5):
                for queue in queues:
                    queue.put_nowait(value)
            notify_many(loop, queues * 2)

        await asyncio.to_thread(produce)
        results = [
            [await asyncio.wait_for(q.get(), 5) for _ in range(5)] for q in queues
        ]
        assert results == [list(range(5)), list(range(5))]

    asyncio.run(run())


def test_notify_raises_on_stopped_loop() -> None:
    loop = asyncio.new_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    try:
        with pytest.raises(EventLoopShutdownError):
            notify_one(loop, queue)
        with pytest.raises(EventLoopShutdownError):
            notify_many(loop, [queue])
    finally:
        loop.close()


def test_notify_raises_on_closed_loop() -> None:
    loop = asyncio.new_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    loop.close()
    with pytest.raises(EventLoopShutdownError):
        notify_one(loop, queue)
    with pytest.raises(EventLoopShutdownError):
        notify_many(loop, [queue])
