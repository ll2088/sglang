"""Stream error events must fail benchmark requests, not count as success."""

import json

import pytest

from sglang.multimodal_gen.benchmarks.bench_serving import (
    _classify_stream_payload,
    _extract_stream_error,
    _finalize_stream_output,
    _stream_error_from_chunk,
)
from sglang.multimodal_gen.benchmarks.datasets import RequestFuncOutput

ERROR_MESSAGE = "internal server error during generation"


def _sse(payload) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _token_chunk(text: str = "tok") -> bytes:
    return _sse({"choices": [{"text": text}]})


def _error_chunk(message: str = ERROR_MESSAGE) -> bytes:
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


def test_extract_stream_error_returns_json_for_object():
    error = _extract_stream_error({"error": {"message": "boom", "code": 500}})
    assert error is not None
    assert "boom" in error


def test_extract_stream_error_returns_string_as_is():
    assert _extract_stream_error({"error": "boom"}) == "boom"


def test_extract_stream_error_ignores_clean_payload():
    assert _extract_stream_error({"choices": [{"text": "a"}]}) is None
    assert _extract_stream_error(None) is None


def test_stream_error_from_chunk_detects_error_after_tokens():
    chunk = _token_chunk("a") + _token_chunk("b") + _error_chunk()
    error = _stream_error_from_chunk(chunk)
    assert error is not None
    assert ERROR_MESSAGE in error


def test_stream_error_from_chunk_ignores_clean_stream():
    chunk = _token_chunk("a") + _token_chunk("b") + b"data: [DONE]\n\n"
    assert _stream_error_from_chunk(chunk) is None


def test_classify_stream_payload_error_wins_over_output():
    has_output, error = _classify_stream_payload(
        {"choices": [{"text": "a"}], "error": {"message": "boom"}}
    )
    assert has_output is False
    assert error is not None and "boom" in error


def test_classify_stream_payload_clean_output():
    has_output, error = _classify_stream_payload({"choices": [{"text": "a"}]})
    assert has_output is True
    assert error is None


def test_finalize_stream_output_error_after_tokens_is_failure():
    output = RequestFuncOutput()
    _finalize_stream_output(output, ERROR_MESSAGE, first_chunk_received=True)
    assert output.success is False
    assert ERROR_MESSAGE in output.error


def test_finalize_stream_output_clean_stream_is_success():
    output = RequestFuncOutput()
    _finalize_stream_output(output, None, first_chunk_received=True)
    assert output.success is True


def test_finalize_stream_output_no_chunk_is_failure():
    output = RequestFuncOutput()
    _finalize_stream_output(output, None, first_chunk_received=False)
    assert output.success is False
    assert "Never received a valid chunk" in output.error


@pytest.mark.parametrize("message", [ERROR_MESSAGE, "engine died"])
def test_error_message_is_preserved(message):
    output = RequestFuncOutput()
    _finalize_stream_output(output, message, first_chunk_received=True)
    assert output.error == message
