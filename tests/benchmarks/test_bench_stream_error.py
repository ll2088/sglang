"""CPU-only tests for stream error detection in benchmark clients.

These tests load the benchmark modules by path so they do not require the
full sglang runtime (torch, kernels, etc.) to be installed.
"""

import importlib.util
import json
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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench_serving():
    pytest.importorskip("aiohttp")
    pytest.importorskip("numpy")
    pytest.importorskip("requests")
    pytest.importorskip("tqdm")
    return _load_module("bench_serving_under_test", BENCH_SERVING)


def _sse(payload) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def test_extract_stream_error_from_object(bench_serving):
    payload = {"error": {"message": "boom", "code": 500}}
    error = bench_serving._extract_stream_error(payload)
    assert error is not None
    assert "boom" in error


def test_extract_stream_error_from_string(bench_serving):
    assert (
        bench_serving._extract_stream_error({"error": "boom"}) == "boom"
    )


def test_extract_stream_error_absent(bench_serving):
    assert bench_serving._extract_stream_error({"choices": []}) is None
    assert bench_serving._extract_stream_error(None) is None


def test_stream_error_from_chunk_after_tokens(bench_serving):
    chunk = (
        _sse({"choices": [{"text": "a"}]})
        + _sse({"choices": [{"text": "b"}]})
        + _sse({"error": {"message": "boom", "code": 500}})
        + b"data: [DONE]\n\n"
    )
    error = bench_serving._stream_error_from_chunk(chunk)
    assert error is not None
    assert "boom" in error


def test_stream_error_from_chunk_clean(bench_serving):
    chunk = (
        _sse({"choices": [{"text": "a"}]})
        + _sse({"choices": [{"text": "b"}]})
        + b"data: [DONE]\n\n"
    )
    assert bench_serving._stream_error_from_chunk(chunk) is None


def test_classify_stream_payload_error_wins(bench_serving):
    has_output, error = bench_serving._classify_stream_payload(
        {"choices": [{"text": "a"}], "error": {"message": "boom"}}
    )
    assert has_output is False
    assert error is not None
    assert "boom" in error


def test_classify_stream_payload_output(bench_serving):
    has_output, error = bench_serving._classify_stream_payload(
        {"choices": [{"text": "a"}]}
    )
    assert has_output is True
    assert error is None


def test_finalize_stream_output_error_after_tokens(bench_serving):
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(
        output, "boom", first_chunk_received=True
    )
    assert output.success is False
    assert output.error == "boom"


def test_finalize_stream_output_clean(bench_serving):
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(
        output, None, first_chunk_received=True
    )
    assert output.success is True
    assert output.error == ""


def test_finalize_stream_output_no_chunk(bench_serving):
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(
        output, None, first_chunk_received=False
    )
    assert output.success is False
    assert "Never received" in output.error
