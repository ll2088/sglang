"""SSE chunk building utilities for OpenAI chat completions streaming."""

from __future__ import annotations

import json
from typing import List, Optional, Union

import msgspec

_SSE_DATA_B = b"data: "
_SSE_NL_B = b"\n\n"


class StreamDelta(msgspec.Struct, omit_defaults=True):
    """Delta content for streaming responses.

    OpenAI Python SDK's ChoiceDelta does not declare reasoning_content; it is
    surfaced via pydantic `extra`. With omit_defaults=True, defaulting to
    None would drop the key entirely from the SSE payload, making
    `data.reasoning_content` raise AttributeError on the client. Keep it
    required (no default) so it is always serialized as null or a string.
    """

    reasoning_content: Optional[str]
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(msgspec.Struct):
    """A single choice in a streaming response."""

    index: int
    delta: StreamDelta
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None
    matched_stop: Union[None, int, str] = None


class StreamChunk(msgspec.Struct, omit_defaults=True):
    """A complete streaming chunk."""

    id: str
    object: str
    created: int
    model: str
    choices: List[StreamChoice]
    usage: Optional[dict] = None


_stream_encoder = msgspec.json.Encoder()

_COMPLETION_PROBE_TEXT = "zZ_sglang_sse_probe_Zz"
_COMPLETION_PROBE_INDEX = 987654321


def build_completion_delta_frame_template(
    chunk_id: str,
    created: int,
    model: str,
    response_cls,
    choice_cls,
) -> Optional[tuple[str, str, str]]:
    """Split a probe completion chunk into the parts of a plain-delta SSE
    frame that are constant for a stream, so only index and text are rendered
    per chunk.

    The template is derived from the real serializer so frames stay
    byte-identical to ``model_dump_json()`` if the models gain fields.

    Returns:
        ``(prefix, infix, suffix)``, or None if the sentinels are ambiguous.
    """
    probe = response_cls(
        id=chunk_id,
        created=created,
        object="text_completion",
        choices=[
            choice_cls(
                index=_COMPLETION_PROBE_INDEX,
                text=_COMPLETION_PROBE_TEXT,
                logprobs=None,
                finish_reason=None,
                matched_stop=None,
                token_ids=None,
                prompt_token_ids=None,
            )
        ],
        model=model,
    )
    frame = f"data: {probe.model_dump_json()}\n\n"
    index = str(_COMPLETION_PROBE_INDEX)
    # Must match the rendering path below, which uses ensure_ascii=False.
    text = json.dumps(_COMPLETION_PROBE_TEXT, ensure_ascii=False)
    if frame.count(index) != 1 or frame.count(text) != 1:
        return None
    prefix, rest = frame.split(index, 1)
    infix, suffix = rest.split(text, 1)
    return prefix, infix, suffix


def build_sse_content(
    chunk_id: str,
    created: int,
    model: str,
    index: int,
    role: Optional[str] = None,
    content: Optional[str] = None,
    reasoning_content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    logprobs: Optional[dict] = None,
    matched_stop: Union[None, int, str] = None,
    usage: Optional[dict] = None,
) -> str:
    """Build an SSE chunk string for content/reasoning updates.

    Args:
        chunk_id: Request ID for this chunk
        created: Unix timestamp
        model: Model name
        index: Choice index
        role: Message role (usually "assistant")
        content: Text content delta
        reasoning_content: Reasoning/thinking content delta
        finish_reason: Finish reason if done
        logprobs: Log probabilities if requested
        matched_stop: Stop token/string that was matched
        usage: Token usage statistics

    Returns:
        SSE-formatted string "data: {...}\\n\\n"
    """
    delta = StreamDelta(role=role, content=content, reasoning_content=reasoning_content)
    choice = StreamChoice(
        index=index,
        delta=delta,
        logprobs=logprobs,
        finish_reason=finish_reason,
        matched_stop=matched_stop,
    )
    chunk = StreamChunk(
        id=chunk_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[choice],
        usage=usage,
    )
    return (_SSE_DATA_B + _stream_encoder.encode(chunk) + _SSE_NL_B).decode()
