import asyncio
import threading

import pytest


class _FakeAsyncQueue:
    def __init__(self):
        self._loop = asyncio.get_event_loop()
        self._queue = asyncio.Queue()
        self._event = asyncio.Event()

    def notify(self):
        self._event.set()

    async def get(self):
        await self._event.wait()
        self._event.clear()
        return await self._queue.get()

    def put_nowait(self, item):
        self._queue.put_nowait(item)


@pytest.mark.parametrize("batched", [False, True])
def test_call_soon_threadsafe_preserves_order(batched):
    async def run():
        loop = asyncio.get_running_loop()
        queues = [_FakeAsyncQueue(), _FakeAsyncQueue()]
        results = {0: [], 1: []}

        async def consume(idx, queue):
            for _ in range(10):
                results[idx].append(await queue.get())

        consumers = [
            asyncio.create_task(consume(i, q)) for i, q in enumerate(queues)
        ]
        await asyncio.sleep(0)

        def produce():
            for value in range(10):
                for q in queues:
                    q.put_nowait(value)
                if batched:
                    loop.call_soon_threadsafe(
                        lambda: [q.notify() for q in queues]
                    )
                else:
                    for q in queues:
                        loop.call_soon_threadsafe(q.notify)

        thread = threading.Thread(target=produce)
        thread.start()
        await asyncio.wait_for(asyncio.gather(*consumers), 5)
        thread.join()
        assert results[0] == list(range(10))
        assert results[1] == list(range(10))

    asyncio.run(run())


def test_call_soon_threadsafe_after_loop_closed_raises():
    loop = asyncio.new_event_loop()
    loop.close()
    with pytest.raises(RuntimeError):
        loop.call_soon_threadsafe(lambda: None)
