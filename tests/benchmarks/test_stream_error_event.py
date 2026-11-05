"""Unit tests for stream-internal error event handling in benchmark clients.

These tests use only the standard library so they can run without installing
heavy serving dependencies. They mirror the upstream rule: a stream that
delivers tokens and then an ``error`` envelope must be reported as failed,
while a clean stream must still be reported as successful.
"""

from __future__ import annotations

import json

import pytest

from sglang.multimodal_gen.benchmarks.bench_serving import (
    _classify_stream_payload,
    _extract_stream_error,
)

_ERROR_MESSAGE = "internal server error during generation"


def _error_payload(message: str = _ERROR_MESSAGE) -> dict:
    return {
        "error": {
            "message": message,
            "type": "InternalServerError",
            "param": None,
            "code": 500,
        }
    }


def _token_payload(text: str = "a") -> dict:
    return {"choices": [{"text": text}]}


def test_extract_stream_error_returns_json_for_object() -> None:
    error = _extract_stream_error(_error_payload())
    assert error is not None
    assert _ERROR_MESSAGE in error
    assert json.loads(error)["code"] == 500


def test_extract_stream_error_returns_string_as_is() -> None:
    assert _extract_stream_error({"error": "boom"}) == "boom"


def test_extract_stream_error_ignores_clean_payload() -> None:
    assert _extract_stream_error(_token_payload()) is None
    assert _extract_stream_error(None) is None
    assert _extract_stream_error("not-a-dict") is None


def test_error_envelope_wins_over_output_in_same_payload() -> None:
    payload = _token_payload()
    payload.update(_error_payload())
    has_output, stream_error = _classify_stream_payload(payload)
    assert has_output is False
    assert stream_error is not None
    assert _ERROR_MESSAGE in stream_error


def test_token_payload_is_output_without_error() -> None:
    has_output, stream_error = _classify_stream_payload(_token_payload())
    assert has_output is True
    assert stream_error is None


def test_stream_tokens_then_error_is_failed() -> None:
    """Two tokens, then the error envelope, then [DONE]."""
    events = [_token_payload("a"), _token_payload("b"), _error_payload()]
    saw_token = False
    stream_error = None
    for payload in events:
        has_output, error = _classify_stream_payload(payload)
        if error is not None:
            stream_error = error
        elif has_output:
            saw_token = True

    assert saw_token
    assert stream_error is not None
    assert _ERROR_MESSAGE in stream_error
    success = stream_error is None and saw_token
    assert not success


def test_clean_stream_is_successful() -> None:
    events = [_token_payload("a"), _token_payload("b")]
    saw_token = False
    stream_error = None
    for payload in events:
        has_output, error = _classify_stream_payload(payload)
        if error is not None:
            stream_error = error
        elif has_output:
            saw_token = True

    assert saw_token
    assert stream_error is None
    assert stream_error is None and saw_token


def test_usage_chunk_does_not_mask_error() -> None:
    events = [
        _token_payload("a"),
        _error_payload(),
        {"usage": {"completion_tokens": 1}},
    ]
    saw_token = False
    stream_error = None
    for payload in events:
        has_output, error = _classify_stream_payload(payload)
        if error is not None:
            stream_error = error
        elif has_output:
            saw_token = True

    assert saw_token
    assert stream_error is not None
    assert not (stream_error is None and saw_token)
