"""call_soon_threadsafe queue notification tests (standard library only)."""

import asyncio
import importlib.util
import pathlib

import pytest


def _load_notify_module():
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "python"
        / "sglang"
        / "srt"
        / "utils"
        / "async_queue_notify.py"
    )
    spec = importlib.util.spec_from_file_location("sglang_async_queue_notify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
 def notify_module():
    return _load_notify_module()


def test_notify_one_from_thread(notify_module):
    async def run():
        loop = asyncio.get_running_loop()
        notifier = notify_module._Notifier(loop)
        waiter = asyncio.create_task(notifier.wait())
        await asyncio.sleep(0)
        await asyncio.to_thread(notify_module.notify_one, loop, notifier)
        await asyncio.wait_for(waiter, 5)

    asyncio.run(run())


def test_notify_many_preserves_order(notify_module):
    async def run():
        loop = asyncio.get_running_loop()
        notifiers = [notify_module._Notifier(loop) for _ in range(3)]
        waiters = [asyncio.create_task(n.wait()) for n in notifiers]
        await asyncio.sleep(0)
        await asyncio.to_thread(notify_module.notify_many, loop, notifiers * 2)
        await asyncio.wait_for(asyncio.gather(*waiters), 5)

    asyncio.run(run())


def test_notify_on_closed_loop_raises(notify_module):
    loop = asyncio.new_event_loop()
    notifier = notify_module._Notifier(loop)
    loop.close()
    with pytest.raises(notify_module.EventLoopShutdownError):
        notify_module.notify_one(loop, notifier)
    with pytest.raises(notify_module.EventLoopShutdownError):
        notify_module.notify_many(loop, [notifier])
