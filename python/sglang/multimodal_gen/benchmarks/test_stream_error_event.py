"""Tests for stream error handling in multimodal generation benchmarks.

A stream that delivers output and then a server error event must be reported
as failed, with the server's error object preserved. A clean stream must still
be reported as successful so the detection does not turn normal runs into
failures.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_BENCH_DIR = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    """Load a benchmark module by path without importing the full package."""
    module_path = _BENCH_DIR / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench_serving = _load_module("bench_serving_stream_error", "bench_serving.py")
request_manifest = _load_module(
    "request_manifest_stream_error", "request_manifest.py"
)

_STREAM_ERROR_MESSAGE = "internal server error during generation"


def _sse(payload) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _token_chunk(text: str) -> bytes:
    return _sse({"choices": [{"text": text}]})


def _error_chunk(message: str = _STREAM_ERROR_MESSAGE) -> bytes:
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


def test_stream_error_after_tokens_is_detected():
    """Tokens followed by an error envelope must surface the server error."""
    chunk = _token_chunk("a") + _token_chunk("b") + _error_chunk()
    assert bench_serving._stream_error_from_chunk(chunk) is not None
    assert _STREAM_ERROR_MESSAGE in bench_serving._stream_error_from_chunk(chunk)


def test_clean_stream_has_no_error():
    """A clean token stream must not be classified as failed."""
    chunk = _token_chunk("a") + _token_chunk("b") + b"data: [DONE]\n\n"
    assert bench_serving._stream_error_from_chunk(chunk) is None


def test_error_wins_over_output_in_same_payload():
    """The server error is the root cause and wins over partial output."""
    payload = {
        "choices": [{"text": "partial"}],
        "error": {"message": _STREAM_ERROR_MESSAGE, "code": 500},
    }
    has_output, stream_error = bench_serving._classify_stream_payload(payload)
    assert not has_output
    assert stream_error is not None
    assert _STREAM_ERROR_MESSAGE in stream_error


def test_finalize_stream_output_marks_error_as_failure():
    """A stream error must fail the request even after tokens arrived."""
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(
        output, _STREAM_ERROR_MESSAGE, first_chunk_received=True
    )
    assert not output.success
    assert _STREAM_ERROR_MESSAGE in output.error


def test_finalize_stream_output_clean_stream_is_success():
    """A clean stream with a first chunk must still be successful."""
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(
        output, None, first_chunk_received=True
    )
    assert output.success


def test_finalize_stream_output_no_chunk_is_failure():
    """No chunk and no error keeps the existing symptom message."""
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(
        output, None, first_chunk_received=False
    )
    assert not output.success
    assert "Never received a valid chunk" in output.error


def test_manifest_rejects_stream_error(tmp_path: Path):
    """A manifest line carrying a stream error must be rejected."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "prompt": "hello",
                "error": {"message": _STREAM_ERROR_MESSAGE, "code": 500},
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="stream error"):
        request_manifest.load_request_manifest(manifest)


def test_manifest_accepts_clean_line(tmp_path: Path):
    """A clean manifest line must still load."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"prompt": "hello"}) + "\n")
    loaded = request_manifest.load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].prompt == "hello"
