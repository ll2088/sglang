"""Tests for stream error detection in multimodal benchmark clients.

A stream that delivers output and then a server ``error`` envelope must be
reported as failed, while a clean stream must still be reported as
successful. These tests do not need a GPU or a running server.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_BENCH_DIR = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    """Load a benchmark module by path without importing the sglang package.

    Importing ``sglang.multimodal_gen.benchmarks`` pulls in the full runtime
    dependency set, which is unnecessary for these pure-Python helpers and
    makes the tests fail in minimal environments. Loading the file directly
    keeps the tests focused on the stream classification logic.
    """
    module_name = f"_stream_error_test_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name, _BENCH_DIR / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_bench_serving = _load_module("bench_serving", "bench_serving.py")
_request_manifest = _load_module("request_manifest", "request_manifest.py")

_classify_stream_payload = _bench_serving._classify_stream_payload
_extract_stream_error = _bench_serving._extract_stream_error
load_request_manifest = _request_manifest.load_request_manifest

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
