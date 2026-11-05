"""Stream error detection tests for the multimodal benchmark clients.

These tests load the benchmark modules by path so they do not import the
``sglang`` package (and therefore do not need the full runtime installed).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCH_DIR = _REPO_ROOT / "python" / "sglang" / "multimodal_gen" / "benchmarks"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench_serving():
    return _load_module("bench_serving_under_test", _BENCH_DIR / "bench_serving.py")


@pytest.fixture(scope="module")
def request_manifest():
    return _load_module(
        "request_manifest_under_test", _BENCH_DIR / "request_manifest.py"
    )


class _Output:
    def __init__(self):
        self.success = False
        self.error = ""


def test_error_envelope_after_tokens_fails(bench_serving):
    payload = {
        "error": {
            "message": "internal server error during generation",
            "type": "InternalServerError",
            "code": 500,
        }
    }
    has_output, stream_error = bench_serving._classify_stream_payload(payload)
    assert has_output is False
    assert "internal server error during generation" in stream_error

    output = _Output()
    bench_serving._finalize_stream_output(output, stream_error, True)
    assert output.success is False
    assert "internal server error during generation" in output.error


def test_clean_stream_is_successful(bench_serving):
    payload = {"choices": [{"text": "hello"}]}
    has_output, stream_error = bench_serving._classify_stream_payload(payload)
    assert has_output is True
    assert stream_error is None

    output = _Output()
    bench_serving._finalize_stream_output(output, stream_error, True)
    assert output.success is True
    assert output.error == ""


def test_no_chunk_reports_symptom(bench_serving):
    output = _Output()
    bench_serving._finalize_stream_output(output, None, False)
    assert output.success is False
    assert "Never received a valid chunk" in output.error


def test_string_error_is_preserved(bench_serving):
    has_output, stream_error = bench_serving._classify_stream_payload(
        {"error": "boom"}
    )
    assert has_output is False
    assert stream_error == "boom"


def test_manifest_rejects_stream_error(request_manifest, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "hi", "error": {"message": "boom"}}) + "\n"
    )
    with pytest.raises(ValueError, match="stream error"):
        request_manifest.load_request_manifest(manifest)


def test_manifest_accepts_clean_request(request_manifest, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"prompt": "hi"}) + "\n")
    loaded = request_manifest.load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].prompt == "hi"
