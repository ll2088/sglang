"""Lightweight tests for stream error detection in the multimodal benchmark client.

These tests load ``bench_serving.py`` by path so they do not import the full
SGLang package (and its heavy runtime dependencies). They cover the rule that a
stream which delivers tokens and then an ``error`` envelope must be reported as
failed, while a clean stream is still reported as successful.
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


class _Output:
    """Minimal stand-in for ``RequestFuncOutput`` used by the helpers."""

    def __init__(self):
        self.success = False
        self.error = ""


@pytest.fixture(scope="module")
def bench_module():
    spec = importlib.util.spec_from_file_location("bench_serving_under_test", BENCH_SERVING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sse(payload) -> bytes:
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


def _token_chunk(text: str) -> bytes:
    return _sse({"choices": [{"text": text, "index": 0}]})


def _error_chunk(message: str = "internal server error during generation") -> bytes:
    return _sse(
        {
            "error": {
                "message": message,
                "type": "InternalServerError",
                "param": None,
                "code": 500,
            }
        }
    )


def test_stream_error_after_tokens_is_detected(bench_module):
    body = _token_chunk("a") + _token_chunk("b") + _error_chunk() + b"data: [DONE]\n\n"
    assert bench_module._extract_stream_error_from_text(body.decode("utf-8")) is not None


def test_stream_error_from_chunk_wins_over_output(bench_module):
    chunk = _token_chunk("a") + _error_chunk()
    error = bench_module._stream_error_from_chunk(chunk)
    assert error is not None
    assert "internal server error" in error


def test_clean_stream_has_no_error(bench_module):
    body = _token_chunk("a") + _token_chunk("b") + b"data: [DONE]\n\n"
    assert bench_module._extract_stream_error_from_text(body.decode("utf-8")) is None


def test_finalize_marks_error_stream_as_failed(bench_module):
    output = _Output()
    bench_module._finalize_stream_output(output, "boom", first_chunk_received=True)
    assert output.success is False
    assert output.error == "boom"


def test_finalize_marks_clean_stream_as_success(bench_module):
    output = _Output()
    bench_module._finalize_stream_output(output, None, first_chunk_received=True)
    assert output.success is True
    assert output.error == ""


def test_finalize_marks_empty_stream_as_failed(bench_module):
    output = _Output()
    bench_module._finalize_stream_output(output, None, first_chunk_received=False)
    assert output.success is False
    assert "Never received a valid chunk" in output.error


def test_json_body_error_is_detected(bench_module):
    body = {"data": [{"url": "http://example/x.png"}, {"error": {"message": "boom"}}]}
    error = bench_module._extract_stream_error_from_response_body(body)
    assert error is not None
    assert "boom" in error


def test_json_body_without_error_is_clean(bench_module):
    body = {"data": [{"url": "http://example/x.png"}]}
    assert bench_module._extract_stream_error_from_response_body(body) is None
