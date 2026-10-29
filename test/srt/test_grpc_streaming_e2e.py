import asyncio

import pytest


class _FakeStream:
    def __init__(self, tokens, cancel_after=None):
        self.tokens = tokens
        self.cancel_after = cancel_after
        self.sent = []
        self.cancelled = False

    async def send(self, token):
        if self.cancel_after is not None and len(self.sent) >= self.cancel_after:
            self.cancelled = True
            raise asyncio.CancelledError()
        self.sent.append(token)


async def _drive(stream, tokens):
    for token in tokens:
        await stream.send(token)


@pytest.mark.asyncio
async def test_streaming_completes_with_full_sequence():
    tokens = [1, 2, 3, 4]
    stream = _FakeStream(tokens)
    await _drive(stream, tokens)
    assert stream.sent == tokens
    assert not stream.cancelled


@pytest.mark.asyncio
async def test_streaming_cancel_midway():
    tokens = [1, 2, 3, 4]
    stream = _FakeStream(tokens, cancel_after=2)
    with pytest.raises(asyncio.CancelledError):
        await _drive(stream, tokens)
    assert stream.sent == [1, 2]
    assert stream.cancelled
