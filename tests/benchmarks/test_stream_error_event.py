"""Tests for stream error event handling in multimodal benchmark clients.

A stream that delivers tokens and then a server ``error`` envelope must be
reported as failed, with the server's error object preserved. Partial output
may be kept for diagnostics but must never count as success.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_BENCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
)


def _load_module(name: str, filename: str):
    """Load a benchmark module by path without importing the sglang package."""
    module_path = _BENCH_DIR / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench_serving():
    return _load_module("bench_serving_under_test", "bench_serving.py")


@pytest.fixture(scope="module")
def request_manifest():
    return _load_module("request_manifest_under_test", "request_manifest.py")


_STREAM_ERROR_MESSAGE = "internal server error during generation"


def _error_payload(message: str = _STREAM_ERROR_MESSAGE) -> dict:
    return {
        "error": {
            "message": message,
            "type": "InternalServerError",
            "param": None,
            "code": 500,
        }
    }


def _token_payload(text: str) -> dict:
    return {"choices": [{"text": text}]}


def test_extract_stream_error_returns_json_for_error_envelope(bench_serving):
    payload = _error_payload()
    extracted = bench_serving._extract_stream_error(payload)
    assert extracted is not None
    assert _STREAM_ERROR_MESSAGE in extracted


def test_extract_stream_error_returns_none_for_token_payload(bench_serving):
    assert bench_serving._extract_stream_error(_token_payload("a")) is None


def test_classify_stream_payload_error_wins_over_output(bench_serving):
    payload = _token_payload("a")
    payload.update(_error_payload())
    has_output, stream_error = bench_serving._classify_stream_payload(payload)
    assert has_output is False
    assert stream_error is not None
    assert _STREAM_ERROR_MESSAGE in stream_error


def test_finalize_stream_output_marks_error_as_failure(bench_serving):
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(
        output, json.dumps(_error_payload()), first_chunk_received=True
    )
    assert output.success is False
    assert _STREAM_ERROR_MESSAGE in output.error


def test_finalize_stream_output_clean_stream_is_success(bench_serving):
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(output, None, first_chunk_received=True)
    assert output.success is True
    assert output.error == ""


def test_finalize_stream_output_no_chunk_reports_symptom(bench_serving):
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(output, None, first_chunk_received=False)
    assert output.success is False
    assert "Never received a valid chunk" in output.error


def test_request_manifest_rejects_top_level_error(request_manifest, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "hello", "error": _error_payload()}) + "\n"
    )
    with pytest.raises(ValueError, match="stream error"):
        request_manifest.load_request_manifest(manifest)


def test_request_manifest_rejects_sampling_params_error(request_manifest, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"prompt": "hello", "sampling_params": {"error": _error_payload()}}
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="stream error"):
        request_manifest.load_request_manifest(manifest)


def test_request_manifest_accepts_clean_request(request_manifest, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"prompt": "hello"}) + "\n")
    loaded = request_manifest.load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].prompt == "hello"
