"""Tests for stream error detection in the multimodal benchmark client.

A stream that delivers tokens and then a server ``error`` event must be
reported as failed, while a clean stream must still be reported as
successful. The helpers under test live in
``sglang.multimodal_gen.benchmarks.bench_serving``.

This file is the single canonical copy of these tests. It is loaded by path
so it does not require the ``sglang`` package to be installed, and it lives
next to the module it exercises to avoid duplicate pytest module names.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_BENCH_SERVING_PATH = Path(__file__).resolve().with_name("bench_serving.py")


def _load_bench_serving():
    """Load bench_serving.py by path so the test needs no installed package."""
    spec = importlib.util.spec_from_file_location(
        "multimodal_bench_serving", _BENCH_SERVING_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench_serving = _load_bench_serving()


class _FakeOutput:
    """Minimal stand-in for RequestFuncOutput used by the finalizer."""

    def __init__(self):
        self.success = False
        self.error = ""


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


def test_error_event_after_tokens_is_detected():
    """Tokens followed by an error envelope must surface the server error."""
    chunk = _token_chunk("a") + _token_chunk("b") + _error_chunk()
    assert bench_serving._stream_error_from_chunk(chunk) is not None
    assert _STREAM_ERROR_MESSAGE in bench_serving._stream_error_from_chunk(chunk)


def test_error_event_marks_stream_failed():
    """A stream that streamed tokens then failed must not be successful."""
    output = _FakeOutput()
    stream_error = bench_serving._stream_error_from_chunk(
        _token_chunk("a") + _token_chunk("b") + _error_chunk()
    )
    bench_serving._finalize_stream_output(output, stream_error, True)
    assert not output.success
    assert _STREAM_ERROR_MESSAGE in output.error


def test_error_takes_precedence_over_no_chunk_symptom():
    """The server error wins over the missing-TTFT symptom message."""
    output = _FakeOutput()
    stream_error = bench_serving._stream_error_from_chunk(_error_chunk())
    bench_serving._finalize_stream_output(output, stream_error, False)
    assert not output.success
    assert _STREAM_ERROR_MESSAGE in output.error
    assert "Never received a valid chunk" not in output.error


def test_clean_stream_is_successful():
    """A clean token stream must still be reported as successful."""
    output = _FakeOutput()
    chunk = _token_chunk("a") + _token_chunk("b") + b"data: [DONE]\n\n"
    stream_error = bench_serving._stream_error_from_chunk(chunk)
    assert stream_error is None
    bench_serving._finalize_stream_output(output, stream_error, True)
    assert output.success
    assert output.error == ""


def test_no_chunk_stream_is_failed():
    """A stream with no valid chunk and no error is still failed."""
    output = _FakeOutput()
    bench_serving._finalize_stream_output(output, None, False)
    assert not output.success
    assert "Never received a valid chunk" in output.error


def test_classify_stream_payload_error_wins():
    """An error envelope wins over output carried by the same payload."""
    has_output, stream_error = bench_serving._classify_stream_payload(
        {"choices": [{"text": "a"}], "error": {"message": "boom"}}
    )
    assert not has_output
    assert stream_error is not None
    assert "boom" in stream_error


def test_classify_stream_payload_clean_output():
    """A payload with choices and no error is classified as output."""
    has_output, stream_error = bench_serving._classify_stream_payload(
        {"choices": [{"text": "a"}]}
    )
    assert has_output
    assert stream_error is None


def test_iter_stream_payloads_skips_done_and_non_json():
    """The [DONE] sentinel and non-JSON lines are skipped."""
    chunk = (
        b"event: message\n"
        + _token_chunk("a")
        + b": comment\n"
        + b"data: [DONE]\n\n"
    )
    payloads = list(bench_serving._iter_stream_payloads(chunk))
    assert payloads == [{"choices": [{"text": "a"}]}]


def test_extract_stream_error_string_and_object():
    """Both string and object error envelopes are normalized to a string."""
    assert bench_serving._extract_stream_error({"error": "boom"}) == "boom"
    as_object = bench_serving._extract_stream_error({"error": {"message": "boom"}})
    assert as_object is not None
    assert "boom" in as_object
    assert bench_serving._extract_stream_error({"choices": []}) is None
    assert bench_serving._extract_stream_error("not-a-dict") is None


def test_error_stream_not_counted_in_metrics():
    """A failed stream must not contribute to completed counts or throughput."""
    ok = _FakeOutput()
    bench_serving._finalize_stream_output(ok, None, True)
    ok.latency = 1.0
    ok.output_count = 1
    ok.peak_memory_mb = 0.0

    failed = _FakeOutput()
    stream_error = bench_serving._stream_error_from_chunk(
        _token_chunk("a") + _error_chunk()
    )
    bench_serving._finalize_stream_output(failed, stream_error, True)
    failed.latency = 0.5
    failed.output_count = 1
    failed.peak_memory_mb = 0.0

    class _Args:
        num_outputs_per_prompt = 1

    metrics = bench_serving.calculate_metrics(
        [ok, failed],
        total_duration=2.0,
        requests_list=[None, None],
        args=_Args(),
        slo_enabled=False,
    )
    assert metrics["completed_requests"] == 1
    assert metrics["failed_requests"] == 1
    assert metrics["completed_outputs"] == 1
    assert metrics["output_throughput_ops"] == pytest.approx(0.5)
