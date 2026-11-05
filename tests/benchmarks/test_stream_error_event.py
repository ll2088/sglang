"""Stream error event handling in benchmark clients.

A generation that fails after the response has started cannot change the
HTTP 200 already sent, so the server reports the failure as an ``error``
envelope inside the stream. These tests pin the rule that such a request is
failed even when tokens arrived first, while a clean stream stays successful.

The modules under test are loaded by path so the tests do not import the
sglang runtime.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_SERVING = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
    / "bench_serving.py"
)
REQUEST_MANIFEST = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
    / "request_manifest.py"
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench_serving() -> ModuleType:
    return _load_module("bench_serving_under_test", BENCH_SERVING)


@pytest.fixture(scope="module")
def request_manifest() -> ModuleType:
    return _load_module("request_manifest_under_test", REQUEST_MANIFEST)


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


def test_extract_stream_error_returns_json_for_object(bench_serving):
    error = bench_serving._extract_stream_error(_error_payload())
    assert error is not None
    assert STREAM_ERROR_MESSAGE in error
    assert json.loads(error)["code"] == 500


def test_extract_stream_error_returns_string_as_is(bench_serving):
    assert (
        bench_serving._extract_stream_error({"error": STREAM_ERROR_MESSAGE})
        == STREAM_ERROR_MESSAGE
    )


def test_extract_stream_error_ignores_clean_payload(bench_serving):
    assert bench_serving._extract_stream_error({"choices": [{"text": "a"}]}) is None
    assert bench_serving._extract_stream_error(None) is None
    assert bench_serving._extract_stream_error("not a dict") is None


def test_classify_stream_payload_error_wins_over_output(bench_serving):
    payload = {
        "choices": [{"text": "partial"}],
        "error": {"message": STREAM_ERROR_MESSAGE, "code": 500},
    }
    has_output, stream_error = bench_serving._classify_stream_payload(payload)
    assert has_output is False
    assert stream_error is not None
    assert STREAM_ERROR_MESSAGE in stream_error


def test_classify_stream_payload_clean_output(bench_serving):
    has_output, stream_error = bench_serving._classify_stream_payload(
        {"choices": [{"text": "a"}]}
    )
    assert has_output is True
    assert stream_error is None


def test_classify_stream_payload_error_only(bench_serving):
    has_output, stream_error = bench_serving._classify_stream_payload(
        _error_payload()
    )
    assert has_output is False
    assert stream_error is not None


def test_manifest_rejects_stream_error_line(request_manifest, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "prompt": "hello",
                "error": {"message": STREAM_ERROR_MESSAGE, "code": 500},
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="stream error"):
        request_manifest.load_request_manifest(manifest)


def test_manifest_accepts_clean_line(request_manifest, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"prompt": "hello"}) + "\n")
    loaded = request_manifest.load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].prompt == "hello"
