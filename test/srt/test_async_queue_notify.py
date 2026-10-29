import asyncio

import pytest

from sglang.srt.utils.async_queue import (
    EventLoopShutdownError,
    notify_many,
    notify_one,
)


class _Queue:
    def __init__(self):
        self.count = 0

    def notify(self):
        self.count += 1


def test_notify_one_from_thread():
    async def run():
        loop = asyncio.get_running_loop()
        queue = _Queue()
        await asyncio.to_thread(notify_one, loop, queue)
        await asyncio.sleep(0)
        assert queue.count == 1

    asyncio.run(run())


def test_notify_many_deduplicates_and_orders():
    async def run():
        loop = asyncio.get_running_loop()
        queues = [_Queue(), _Queue()]
        await asyncio.to_thread(notify_many, loop, queues + queues)
        await asyncio.sleep(0)
        assert [q.count for q in queues] == [1, 1]

    asyncio.run(run())


def test_notify_raises_on_stopped_loop():
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(EventLoopShutdownError):
            notify_one(loop, _Queue())
        with pytest.raises(EventLoopShutdownError):
            notify_many(loop, [_Queue()])
    finally:
        loop.close()


def test_notify_raises_on_closed_loop():
    loop = asyncio.new_event_loop()
    loop.close()
    with pytest.raises(EventLoopShutdownError):
        notify_one(loop, _Queue())
    with pytest.raises(EventLoopShutdownError):
        notify_many(loop, [_Queue()])
