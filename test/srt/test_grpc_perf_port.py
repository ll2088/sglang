import asyncio
import builtins
import importlib
import os
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Protobuf implementation detection (mirrors sglang.srt.utils.protobuf_impl)
# ---------------------------------------------------------------------------


def _warn_if_python_protobuf(logger):
    """Local copy of the detection logic so the test does not import sglang."""
    try:
        from google.protobuf.internal import api_implementation

        implementation = api_implementation.Type()
    except (ImportError, AttributeError):
        return
    if implementation == "python":
        logger.warning(
            "SGLang gRPC is using Python protobuf, which can reduce throughput. "
            "Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb before starting "
            "Python and install a supported protobuf wheel with the upb "
            "extension. Changing the environment after importing protobuf "
            "does not change the active implementation."
        )


class _FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


@pytest.mark.parametrize("implementation", ["python", "upb"])
def test_protobuf_warning_in_fresh_process(implementation):
    script = (
        "import os, sys\n"
        "from unittest.mock import patch\n"
        "from google.protobuf.internal import api_implementation\n"
        "import test.srt.test_grpc_perf_port as mod\n"
        "assert api_implementation.Type() == sys.argv[1]\n"
        "os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "
        "'upb' if sys.argv[1] == 'python' else 'python'\n"
        "logger = mod._FakeLogger()\n"
        "mod._warn_if_python_protobuf(logger)\n"
        "if sys.argv[1] == 'python':\n"
        "    assert len(logger.warnings) == 1\n"
        "    assert 'before starting Python' in logger.warnings[0]\n"
        "else:\n"
        "    assert logger.warnings == []\n"
        "assert api_implementation.Type() == sys.argv[1]\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, implementation],
        env={**os.environ, "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": implementation},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cpp_protobuf_does_not_warn(monkeypatch):
    from google.protobuf.internal import api_implementation

    logger = _FakeLogger()
    monkeypatch.setattr(api_implementation, "Type", lambda: "cpp")
    _warn_if_python_protobuf(logger)
    assert logger.warnings == []


@pytest.mark.parametrize("missing_api", ["import", "Type"])
def test_protobuf_warning_skips_unavailable_api(monkeypatch, missing_api):
    from google.protobuf.internal import api_implementation

    logger = _FakeLogger()
    if missing_api == "import":
        real_import = builtins.__import__

        def import_without_implementation(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            if name == "google.protobuf.internal" and "api_implementation" in fromlist:
                raise ImportError("protobuf implementation API unavailable")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", import_without_implementation)
    else:
        monkeypatch.delattr(api_implementation, "Type")

    _warn_if_python_protobuf(logger)
    assert logger.warnings == []


# ---------------------------------------------------------------------------
# Streaming token history zero-copy
# ---------------------------------------------------------------------------


def _chunk_responses(result, sent):
    """Reference implementation of the zero-copy streaming chunker."""
    responses = []
    for output in result.outputs:
        token_ids = output.token_ids
        if not token_ids:
            continue
        start = sent.get(output.index, 0)
        if start >= len(token_ids):
            continue
        delta = token_ids[start:]
        logprobs = getattr(output, "logprobs", None)
        delta_logprobs = logprobs[start:] if logprobs is not None else None
        sent[output.index] = len(token_ids)
        responses.append(
            SimpleNamespace(
                sequence_index=output.index,
                token_ids=delta,
                logprobs=delta_logprobs,
                completion_tokens=len(token_ids),
                prompt_tokens=result.prompt_tokens,
                cached_tokens=getattr(result, "cached_tokens", 0),
            )
        )
    return responses


def test_streaming_deltas_preserve_sequences_logprobs_and_final_tokens():
    first = SimpleNamespace(index=0, token_ids=[10], logprobs=[-0.5], finish_reason=None)
    second = SimpleNamespace(
        index=1, token_ids=[20, 21], logprobs=[-1.0, -1.5], finish_reason=None
    )
    result = SimpleNamespace(outputs=[first, second], cached_tokens=2, prompt_tokens=3)
    sent = {}
    initial = _chunk_responses(result, sent)
    initial_bytes = [repr(r) for r in initial]
    assert [list(r.token_ids) for r in initial] == [[10], [20, 21]]
    assert sent == {0: 1, 1: 2}

    first.token_ids.append(11)
    first.logprobs.append(-0.25)
    delta = _chunk_responses(result, sent)
    assert len(delta) == 1
    assert delta[0].sequence_index == 0
    assert list(delta[0].token_ids) == [11]
    assert delta[0].completion_tokens == 2
    assert delta[0].prompt_tokens == 3
    assert delta[0].cached_tokens == 2
    assert list(delta[0].logprobs) == [-0.25]
    assert [repr(r) for r in initial] == initial_bytes
    assert _chunk_responses(result, sent) == []


def test_streaming_zero_copy_does_not_mutate_source():
    token_ids = [1, 2, 3]
    logprobs = [-0.1, -0.2, -0.3]
    output = SimpleNamespace(index=0, token_ids=token_ids, logprobs=logprobs)
    result = SimpleNamespace(outputs=[output], cached_tokens=0, prompt_tokens=1)
    sent = {}
    responses = _chunk_responses(result, sent)
    assert len(responses) == 1
    assert list(responses[0].token_ids) == [1, 2, 3]
    assert token_ids == [1, 2, 3]
    assert logprobs == [-0.1, -0.2, -0.3]


@pytest.mark.parametrize("token_ids", [None, []])
def test_streaming_empty_token_history(token_ids):
    result = SimpleNamespace(
        outputs=[SimpleNamespace(index=0, token_ids=token_ids)],
        cached_tokens=0,
        prompt_tokens=0,
    )
    sent = {}
    assert _chunk_responses(result, sent) == []
    assert sent == {}


# ---------------------------------------------------------------------------
# Queue notification via call_soon_threadsafe
# ---------------------------------------------------------------------------


class _AsyncQueue:
    """Minimal asyncio queue wrapper mirroring the upstream _SyncQueue."""

    class EventLoopShutdownError(RuntimeError):
        pass

    def __init__(self, loop=None):
        self._loop = loop or asyncio.get_event_loop()
        self._queue = asyncio.Queue()
        self._event = asyncio.Event()

    def put(self, item):
        self._queue.put_nowait(item)
        if not self._loop.is_running():
            raise _AsyncQueue.EventLoopShutdownError()
        try:
            self._loop.call_soon_threadsafe(self._event.set)
        except RuntimeError as error:
            if self._loop.is_closed():
                raise _AsyncQueue.EventLoopShutdownError() from error
            raise

    async def get(self):
        await self._event.wait()
        self._event.clear()
        return self._queue.get_nowait()

    def empty(self):
        return self._queue.empty()


@pytest.mark.parametrize("batched", [False, True])
def test_async_queue_thread_notifications(batched):
    async def run():
        queues = [_AsyncQueue(), _AsyncQueue()]

        async def consume(queue):
            return [await queue.get() for _ in range(10)]

        def produce(offset):
            for value in range(offset, offset + 10):
                for queue in queues:
                    queue.put(value)

        for offset in (0, 10, 20):
            consumers = [asyncio.create_task(consume(q)) for q in queues]
            await asyncio.sleep(0)
            await asyncio.to_thread(produce, offset)
            values = await asyncio.wait_for(asyncio.gather(*consumers), 5)
            assert values == [list(range(offset, offset + 10))] * 2
            assert all(q.empty() for q in queues)

    asyncio.run(run())


@pytest.mark.parametrize("state", ["stopped", "closed", "close_during_schedule"])
def test_async_queue_notification_shutdown(state, monkeypatch):
    loop = asyncio.new_event_loop()

    async def make_queue():
        return _AsyncQueue(loop)

    queue = loop.run_until_complete(make_queue())
    if state == "closed":
        loop.close()
    elif state == "close_during_schedule":
        monkeypatch.setattr(loop, "is_running", lambda: True)

        def close(*args):
            monkeypatch.setattr(loop, "is_running", lambda: False)
            loop.close()
            raise RuntimeError("Event loop is closed")

        monkeypatch.setattr(loop, "call_soon_threadsafe", close)
    try:
        with pytest.raises(_AsyncQueue.EventLoopShutdownError):
            queue.put(1)
    finally:
        loop.close()
