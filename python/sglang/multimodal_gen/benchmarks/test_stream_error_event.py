"""Tests for stream error detection in multimodal benchmark clients.

A stream that delivers output and then a server ``error`` envelope must be
reported as failed, while a clean stream must still be reported as
successful. These tests do not need a GPU or a running server.
"""

import json

import pytest

from sglang.multimodal_gen.benchmarks.bench_serving import (
    _classify_stream_payload,
    _extract_stream_error,
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


def test_extract_stream_error_returns_json_for_object() -> None:
    payload = _error_payload()
    error = _extract_stream_error(payload)
    assert error is not None
    assert STREAM_ERROR_MESSAGE in error
    assert json.loads(error) == payload["error"]


def test_extract_stream_error_returns_string_as_is() -> None:
    assert _extract_stream_error({"error": STREAM_ERROR_MESSAGE}) == (
        STREAM_ERROR_MESSAGE
    )


def test_extract_stream_error_ignores_clean_payload() -> None:
    assert _extract_stream_error({"choices": [{"text": "a"}]}) is None
    assert _extract_stream_error(None) is None


def test_error_after_tokens_is_not_success() -> None:
    """Tokens followed by an error envelope must be classified as failed."""
    token_payload = {"choices": [{"text": "a"}]}
    has_output, stream_error = _classify_stream_payload(token_payload)
    assert has_output
    assert stream_error is None

    has_output, stream_error = _classify_stream_payload(_error_payload())
    assert not has_output
    assert stream_error is not None
    assert STREAM_ERROR_MESSAGE in stream_error

    # The benchmark rule: an error envelope wins over tokens.
    success = stream_error is None and has_output
    assert not success


def test_clean_stream_is_successful() -> None:
    """Guard against over-eager error detection on a clean stream."""
    payloads = [
        {"choices": [{"text": "a"}]},
        {"choices": [{"text": "b"}]},
        {"data": [{"b64_json": "..."}]},
    ]
    saw_output = False
    stream_error = None
    for payload in payloads:
        has_output, error = _classify_stream_payload(payload)
        saw_output = saw_output or has_output
        stream_error = stream_error or error

    assert saw_output
    assert stream_error is None
    assert stream_error is None and saw_output


def test_manifest_rejects_stream_error(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "prompt": "a cat",
                "error": {"message": STREAM_ERROR_MESSAGE, "code": 500},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stream error"):
        load_request_manifest(manifest)


def test_manifest_accepts_null_error(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "a cat", "error": None}) + "\n",
        encoding="utf-8",
    )
    loaded = load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].prompt == "a cat"
