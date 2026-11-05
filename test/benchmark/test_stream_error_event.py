"""Unit tests for stream error event handling in benchmark clients.

These tests are intentionally dependency-light (stdlib only) so they can run
without installing the full sglang runtime. They cover the rule that a stream
which delivered tokens and then an ``error`` envelope must be reported as a
failure, while a clean stream must still be reported as a success.
"""

import importlib.util
import json
import sys
from pathlib import Path

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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench_serving():
    return _load_module("sglang_bench_serving_under_test", BENCH_SERVING)


@pytest.fixture(scope="module")
def request_manifest():
    return _load_module("sglang_request_manifest_under_test", REQUEST_MANIFEST)


ERROR_ENVELOPE = {
    "error": {
        "message": "internal server error during generation",
        "type": "InternalServerError",
        "param": None,
        "code": 500,
    }
}


def test_extract_stream_error_returns_json_for_object(bench_serving):
    extracted = bench_serving._extract_stream_error(ERROR_ENVELOPE)
    assert extracted is not None
    assert "internal server error during generation" in extracted
    assert json.loads(extracted)["code"] == 500


def test_extract_stream_error_returns_string_as_is(bench_serving):
    assert bench_serving._extract_stream_error({"error": "boom"}) == "boom"


def test_extract_stream_error_none_for_clean_payload(bench_serving):
    assert bench_serving._extract_stream_error({"choices": [{"text": "a"}]}) is None
    assert bench_serving._extract_stream_error(None) is None
    assert bench_serving._extract_stream_error("not-a-dict") is None


def test_classify_error_wins_over_output(bench_serving):
    payload = {
        "choices": [{"text": "partial"}],
        "error": {"message": "boom", "code": 500},
    }
    has_output, stream_error = bench_serving._classify_stream_payload(payload)
    assert has_output is False
    assert stream_error is not None
    assert "boom" in stream_error


def test_classify_clean_token_payload(bench_serving):
    has_output, stream_error = bench_serving._classify_stream_payload(
        {"choices": [{"text": "a"}]}
    )
    assert has_output is True
    assert stream_error is None


def test_classify_clean_image_payload(bench_serving):
    has_output, stream_error = bench_serving._classify_stream_payload(
        {"data": [{"b64_json": "..."}]}
    )
    assert has_output is True
    assert stream_error is None


def test_classify_error_only_payload(bench_serving):
    has_output, stream_error = bench_serving._classify_stream_payload(ERROR_ENVELOPE)
    assert has_output is False
    assert stream_error is not None


def test_manifest_rejects_error_field(request_manifest, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "prompt": "hello",
                "error": {"message": "boom", "code": 500},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stream error"):
        request_manifest.load_request_manifest(manifest)


def test_manifest_accepts_clean_request(request_manifest, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "hello", "request_id": "r1"}) + "\n",
        encoding="utf-8",
    )
    loaded = request_manifest.load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].request_id == "r1"
