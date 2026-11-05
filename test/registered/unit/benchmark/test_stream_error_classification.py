"""Unit tests for stream error classification in the multimodal benchmark client.

These tests exercise the pure-Python SSE helpers in
``sglang.multimodal_gen.benchmarks.bench_serving`` so the rule "a stream that
delivered tokens and then a server error event must fail" is verified without
aiohttp, a server, or a GPU.
"""

import json

import pytest

from sglang.multimodal_gen.benchmarks.bench_serving import (
    _classify_stream_payload,
    _extract_stream_error,
    _extract_stream_error_from_text,
    _finalize_stream_output,
    _stream_error_from_chunk,
)
from sglang.multimodal_gen.benchmarks.datasets import RequestFuncOutput


def _sse(payload) -> bytes:
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


def _token_chunk(text: str) -> bytes:
    return _sse({"choices": [{"text": text}]})


def _error_chunk(message: str = "internal server error") -> bytes:
    return _sse(
        {
            "error": {
                "message": message,
                "type": "InternalServerError",
                "param": None,
                "code": 500,
            }
        }
    )


def test_error_after_tokens_is_detected_in_chunk() -> None:
    body = _token_chunk("a") + _token_chunk("b") + _error_chunk("boom")
    stream_error = _stream_error_from_chunk(body)
    assert stream_error is not None
    assert "boom" in stream_error


def test_error_after_tokens_is_detected_in_text() -> None:
    body = (
        _token_chunk("a") + _token_chunk("b") + _error_chunk("boom")
    ).decode("utf-8")
    stream_error = _extract_stream_error_from_text(body)
    assert stream_error is not None
    assert "boom" in stream_error


def test_clean_stream_has_no_error() -> None:
    body = _token_chunk("a") + _token_chunk("b") + b"data: [DONE]\n\n"
    assert _stream_error_from_chunk(body) is None
    assert _extract_stream_error_from_text(body.decode("utf-8")) is None


def test_string_error_is_supported() -> None:
    assert _extract_stream_error({"error": "boom"}) == "boom"
    assert _extract_stream_error({"choices": []}) is None
    assert _extract_stream_error("not-a-dict") is None


def test_error_wins_over_output_in_same_payload() -> None:
    has_output, stream_error = _classify_stream_payload(
        {"choices": [{"text": "a"}], "error": {"message": "boom"}}
    )
    assert has_output is False
    assert stream_error is not None
    assert "boom" in stream_error


def test_token_payload_is_classified_as_output() -> None:
    has_output, stream_error = _classify_stream_payload(
        {"choices": [{"text": "a"}]}
    )
    assert has_output is True
    assert stream_error is None


@pytest.mark.parametrize(
    "stream_error,first_chunk_received,expected_success",
    [
        (None, True, True),
        (None, False, False),
        ("boom", True, False),
        ("boom", False, False),
    ],
)
def test_finalize_stream_output(
    stream_error, first_chunk_received, expected_success
) -> None:
    output = RequestFuncOutput()
    _finalize_stream_output(output, stream_error, first_chunk_received)
    assert output.success is expected_success
    if stream_error is not None:
        assert output.error == stream_error
    elif not first_chunk_received:
        assert output.error
