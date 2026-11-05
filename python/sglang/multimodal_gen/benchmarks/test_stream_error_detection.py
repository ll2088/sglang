"""Unit tests for stream error detection in the multimodal benchmark client.

These tests exercise the pure classification helpers in ``bench_serving`` so
that a stream which delivers tokens and then a server ``error`` envelope is
reported as failed, while a clean stream is still reported as successful.
"""

import json

from sglang.multimodal_gen.benchmarks.bench_serving import (
    _classify_stream_payload,
    _extract_stream_error,
    _finalize_stream_output,
)
from sglang.multimodal_gen.benchmarks.datasets import RequestFuncOutput

_ERROR_MESSAGE = "internal server error during generation"


def _error_payload(message: str = _ERROR_MESSAGE) -> dict:
    return {
        "error": {
            "message": message,
            "type": "InternalServerError",
            "param": None,
            "code": 500,
        }
    }


def test_extract_stream_error_returns_json_for_object():
    payload = _error_payload()
    extracted = _extract_stream_error(payload)
    assert extracted is not None
    assert _ERROR_MESSAGE in extracted
    assert json.loads(extracted) == payload["error"]


def test_extract_stream_error_returns_string_as_is():
    assert _extract_stream_error({"error": "boom"}) == "boom"


def test_extract_stream_error_ignores_clean_payload():
    assert _extract_stream_error({"choices": [{"text": "a"}]}) is None
    assert _extract_stream_error(None) is None
    assert _extract_stream_error("not-a-dict") is None


def test_classify_error_wins_over_output():
    payload = {
        "choices": [{"text": "partial"}],
        "error": {"message": _ERROR_MESSAGE, "code": 500},
    }
    has_output, stream_error = _classify_stream_payload(payload)
    assert has_output is False
    assert stream_error is not None
    assert _ERROR_MESSAGE in stream_error


def test_classify_clean_token_payload():
    has_output, stream_error = _classify_stream_payload(
        {"choices": [{"text": "hello"}]}
    )
    assert has_output is True
    assert stream_error is None


def test_finalize_error_after_tokens_is_failure():
    output = RequestFuncOutput()
    output.generated_text = "partial"
    _finalize_stream_output(output, _ERROR_MESSAGE, first_chunk_received=True)
    assert output.success is False
    assert _ERROR_MESSAGE in output.error
    # Partial output is kept for detailed saving.
    assert output.generated_text == "partial"


def test_finalize_clean_stream_is_success():
    output = RequestFuncOutput()
    _finalize_stream_output(output, None, first_chunk_received=True)
    assert output.success is True
    assert output.error == ""


def test_finalize_no_chunk_reports_symptom():
    output = RequestFuncOutput()
    _finalize_stream_output(output, None, first_chunk_received=False)
    assert output.success is False
    assert "Never received a valid chunk" in output.error


def test_error_takes_precedence_over_no_chunk_symptom():
    output = RequestFuncOutput()
    _finalize_stream_output(output, _ERROR_MESSAGE, first_chunk_received=False)
    assert output.success is False
    assert _ERROR_MESSAGE in output.error
    assert "Never received a valid chunk" not in output.error
