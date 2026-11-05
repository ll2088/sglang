"""Unit tests for in-stream server error detection in benchmark clients.

These tests load the benchmark modules by file path so that pytest collection
does not import the ``sglang`` top-level package (and therefore does not need
its heavy runtime dependencies installed).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCH_SERVING = (
    _REPO_ROOT
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
    / "bench_serving.py"
)
_REQUEST_MANIFEST = (
    _REPO_ROOT
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
    / "request_manifest.py"
)

_STREAM_ERROR_MESSAGE = "internal server error during generation"


def _load_module_by_path(name: str, path: Path) -> ModuleType:
    """Load a module from an explicit file path without importing its package."""
    if not path.is_file():
        pytest.skip(f"benchmark module not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.skip(f"cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
 def bench_serving_module() -> ModuleType:
    return _load_module_by_path("sglang_bench_serving_under_test", _BENCH_SERVING)


@pytest.fixture(scope="module")
 def request_manifest_module() -> ModuleType:
    return _load_module_by_path(
        "sglang_request_manifest_under_test", _REQUEST_MANIFEST
    )


def _error_envelope(message: str = _STREAM_ERROR_MESSAGE) -> dict:
    """The error object the server sends inside an HTTP 200 stream."""
    return {
        "error": {
            "message": message,
            "type": "InternalServerError",
            "param": None,
            "code": 500,
        }
    }


def test_extract_stream_error_returns_serialized_error(bench_serving_module):
    payload = _error_envelope()
    extracted = bench_serving_module._extract_stream_error(payload)
    assert extracted is not None
    assert _STREAM_ERROR_MESSAGE in extracted
    assert json.loads(extracted) == payload["error"]


def test_extract_stream_error_accepts_plain_string(bench_serving_module):
    assert (
        bench_serving_module._extract_stream_error({"error": "boom"}) == "boom"
    )


def test_extract_stream_error_ignores_clean_payload(bench_serving_module):
    assert bench_serving_module._extract_stream_error({"choices": [{"text": "a"}]}) is None
    assert bench_serving_module._extract_stream_error(None) is None
    assert bench_serving_module._extract_stream_error("not-a-dict") is None


def test_error_envelope_wins_over_output_in_same_payload(bench_serving_module):
    payload = {
        "choices": [{"text": "partial"}],
        **_error_envelope(),
    }
    has_output, stream_error = bench_serving_module._classify_stream_payload(payload)
    assert has_output is False
    assert stream_error is not None
    assert _STREAM_ERROR_MESSAGE in stream_error


def test_clean_token_payload_is_output_without_error(bench_serving_module):
    has_output, stream_error = bench_serving_module._classify_stream_payload(
        {"choices": [{"text": "a"}]}
    )
    assert has_output is True
    assert stream_error is None


def test_clean_image_payload_is_output_without_error(bench_serving_module):
    has_output, stream_error = bench_serving_module._classify_stream_payload(
        {"data": [{"b64_json": "..."}]}
    )
    assert has_output is True
    assert stream_error is None


def test_stream_that_ends_in_error_is_not_successful(bench_serving_module):
    """Two tokens, then an error envelope, then [DONE] must not be success."""
    events = [
        {"choices": [{"text": "a"}]},
        {"choices": [{"text": "b"}]},
        _error_envelope(),
    ]
    saw_output = False
    stream_error = None
    for payload in events:
        has_output, error = bench_serving_module._classify_stream_payload(payload)
        saw_output = saw_output or has_output
        if error is not None:
            stream_error = error
    success = stream_error is None and saw_output
    assert saw_output, "tokens must have been observed before the error"
    assert stream_error is not None
    assert not success


def test_stream_without_error_is_successful(bench_serving_module):
    events = [
        {"choices": [{"text": "a"}]},
        {"choices": [{"text": "b"}]},
    ]
    saw_output = False
    stream_error = None
    for payload in events:
        has_output, error = bench_serving_module._classify_stream_payload(payload)
        saw_output = saw_output or has_output
        if error is not None:
            stream_error = error
    assert saw_output
    assert stream_error is None
    assert stream_error is None and saw_output


def test_manifest_rejects_record_with_stream_error(request_manifest_module, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "prompt": "hello",
                "error": {"message": _STREAM_ERROR_MESSAGE, "code": 500},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stream error"):
        request_manifest_module.load_request_manifest(manifest)


def test_manifest_accepts_clean_record(request_manifest_module, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "hello", "request_id": "r1"}) + "\n",
        encoding="utf-8",
    )
    loaded = request_manifest_module.load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].request_id == "r1"
