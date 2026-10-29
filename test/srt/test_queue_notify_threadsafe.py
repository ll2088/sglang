"""Stability tests for cross-thread queue notifications."""

import asyncio

from sglang.srt.utils.async_queue import notify_many, notify_one


def test_repeated_cross_thread_notifications() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def produce(offset: int) -> None:
            for value in range(offset, offset + 10):
                queue.put_nowait(value)
                notify_one(loop, queue)

        for offset in (0, 10, 20):
            await asyncio.to_thread(produce, offset)
            received = [await asyncio.wait_for(queue.get(), 5) for _ in range(10)]
            assert received == list(range(offset, offset + 10))
            assert queue.empty()

    asyncio.run(run())


def test_late_batch_notification_does_not_strand_next_get() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        queues = [asyncio.Queue(), asyncio.Queue()]
        await asyncio.to_thread(notify_many, loop, queues)
        await asyncio.sleep(0)
        for queue in queues:
            queue.put_nowait(1)
            assert await asyncio.wait_for(queue.get(), 5) == 1

    asyncio.run(run())
