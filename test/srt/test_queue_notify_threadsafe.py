import asyncio
import threading

import pytest


class _FakeAsyncQueue:
    def __init__(self):
        self.notified = 0

    def notify(self):
        self.notified += 1


class _FakeSyncQueue:
    def __init__(self):
        self._aq = _FakeAsyncQueue()


def _notify_many(queues):
    for queue in queues:
        queue._aq.notify()


def test_call_soon_threadsafe_preserves_order():
    async def run():
        loop = asyncio.get_running_loop()
        queue = _FakeSyncQueue()
        seen = []

        def producer():
            for _ in range(10):
                loop.call_soon_threadsafe(queue._aq.notify)
                seen.append(queue._aq.notified)

        thread = threading.Thread(target=producer)
        thread.start()
        thread.join()
        await asyncio.sleep(0)
        assert queue._aq.notified == 10

    asyncio.run(run())


def test_notify_many_deduplicates_and_orders():
    async def run():
        loop = asyncio.get_running_loop()
        queues = [_FakeSyncQueue(), _FakeSyncQueue()]
        loop.call_soon_threadsafe(_notify_many, frozenset(queues))
        await asyncio.sleep(0)
        assert all(q._aq.notified == 1 for q in queues)

    asyncio.run(run())


def test_notify_after_loop_closed_raises():
    loop = asyncio.new_event_loop()
    loop.close()
    with pytest.raises(RuntimeError):
        loop.call_soon_threadsafe(lambda: None)
