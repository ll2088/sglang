"""Tests for stream error event handling in benchmark clients."""

import json

import pytest

from sglang.multimodal_gen.benchmarks.bench_serving import (
    _extract_stream_error,
    _get_response_output_count,
)
from sglang.multimodal_gen.benchmarks.request_manifest import load_request_manifest


STREAM_ERROR_MESSAGE = "internal server error during generation"


def _error_payload(message: str = STREAM_ERROR_MESSAGE) -> dict:
    return {
        "error": {
            "message": message,
            "type": "InternalServerError",
            "param": None,
            "code": 500,
        }
    }


def test_extract_stream_error_returns_server_error() -> None:
    payload = _error_payload()
    error = _extract_stream_error(payload)
    assert error is not None
    assert STREAM_ERROR_MESSAGE in error


def test_extract_stream_error_ignores_normal_chunks() -> None:
    assert _extract_stream_error({"choices": [{"text": "a"}]}) is None
    assert _extract_stream_error({"choices": [{"text": "b"}]}) is None
    assert _extract_stream_error({}) is None
    assert _extract_stream_error(None) is None


def test_extract_stream_error_accepts_string_error() -> None:
    assert _extract_stream_error({"error": "boom"}) == "boom"


def test_request_manifest_rejects_stream_error(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "hello", "error": _error_payload()["error"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stream error"):
        load_request_manifest(manifest)


def test_request_manifest_accepts_normal_request(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "hello", "sampling_params": {"width": 64}}) + "\n",
        encoding="utf-8",
    )
    loaded = load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].prompt == "hello"


def test_extract_stream_error_ignores_usage_only_chunk() -> None:
    # A trailing usage summary carries no error and must not fail the request.
    assert _extract_stream_error({"usage": {"completion_tokens": 2}}) is None


def test_response_output_count_ignores_error_payload() -> None:
    # An error envelope carries no outputs, so it must not be counted.
    assert _get_response_output_count(_error_payload()) == 0
