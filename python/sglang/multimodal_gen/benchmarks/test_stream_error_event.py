"""Tests for stream error detection in the multimodal serving benchmark."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_bench_serving():
    module_path = Path(__file__).with_name("bench_serving.py")
    spec = importlib.util.spec_from_file_location("bench_serving", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench_serving = _load_bench_serving()


def _sse(payload) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _token_chunk(text: str) -> bytes:
    return _sse({"choices": [{"text": text}]})


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


def test_stream_error_from_chunk_detects_error_envelope():
    chunk = _token_chunk("a") + _error_chunk("boom")
    error = bench_serving._stream_error_from_chunk(chunk)
    assert error is not None
    assert "boom" in error


def test_stream_error_from_chunk_ignores_clean_chunk():
    chunk = _token_chunk("a") + _token_chunk("b") + b"data: [DONE]\n\n"
    assert bench_serving._stream_error_from_chunk(chunk) is None


def test_classify_stream_payload_error_wins_over_output():
    payload = {
        "choices": [{"text": "partial"}],
        "error": {"message": "boom", "code": 500},
    }
    has_output, stream_error = bench_serving._classify_stream_payload(payload)
    assert has_output is False
    assert stream_error is not None
    assert "boom" in stream_error


def test_classify_stream_payload_clean_output():
    has_output, stream_error = bench_serving._classify_stream_payload(
        {"choices": [{"text": "a"}]}
    )
    assert has_output is True
    assert stream_error is None


def test_finalize_stream_output_error_marks_failure():
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(
        output, stream_error="boom", first_chunk_received=True
    )
    assert output.success is False
    assert "boom" in output.error


def test_finalize_stream_output_clean_stream_succeeds():
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(
        output, stream_error=None, first_chunk_received=True
    )
    assert output.success is True


def test_finalize_stream_output_no_chunk_reports_symptom():
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(
        output, stream_error=None, first_chunk_received=False
    )
    assert output.success is False
    assert "Never received a valid chunk" in output.error


def test_error_event_stream_is_not_reported_as_success():
    """Two tokens, then an error envelope, then [DONE] must fail."""
    chunks = [_token_chunk("a"), _token_chunk("b"), _error_chunk(), b"data: [DONE]\n\n"]
    stream_error = None
    first_chunk_received = False
    for chunk in chunks:
        for payload in bench_serving._iter_stream_payloads(chunk):
            has_output, err = bench_serving._classify_stream_payload(payload)
            if err is not None:
                stream_error = err
            elif has_output:
                first_chunk_received = True

    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(output, stream_error, first_chunk_received)
    assert output.success is False
    assert "internal server error during generation" in output.error


def test_clean_stream_is_reported_as_success():
    chunks = [_token_chunk("a"), _token_chunk("b"), b"data: [DONE]\n\n"]
    stream_error = None
    first_chunk_received = False
    for chunk in chunks:
        for payload in bench_serving._iter_stream_payloads(chunk):
            has_output, err = bench_serving._classify_stream_payload(payload)
            if err is not None:
                stream_error = err
            elif has_output:
                first_chunk_received = True

    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(output, stream_error, first_chunk_received)
    assert output.success is True
