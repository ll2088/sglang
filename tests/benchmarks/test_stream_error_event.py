"""Stream error-event handling for the multimodal benchmark clients.

A generation that fails after the HTTP 200 has been sent reports the failure
as an ``error`` envelope inside the stream. Such a request must be counted as
failed even if tokens arrived first, and a clean stream must stay successful.
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
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench_serving = _load_module("bench_serving_stream_error", "bench_serving.py")
request_manifest = _load_module(
    "request_manifest_stream_error", "request_manifest.py"
)


class _Output:
    """Minimal stand-in for ``RequestFuncOutput``."""

    def __init__(self) -> None:
        self.success = False
        self.error = ""


_ERROR_MESSAGE = "internal server error during generation"


def _error_payload(message: str = _ERROR_MESSAGE):
    return {
        "error": {
            "message": message,
            "type": "InternalServerError",
            "param": None,
            "code": 500,
        }
    }


def test_error_envelope_is_extracted():
    error = bench_serving._extract_stream_error(_error_payload())
    assert error is not None
    assert _ERROR_MESSAGE in error


def test_string_error_envelope_is_extracted():
    assert (
        bench_serving._extract_stream_error({"error": _ERROR_MESSAGE})
        == _ERROR_MESSAGE
    )


def test_payload_without_error_has_no_stream_error():
    assert bench_serving._extract_stream_error({"choices": [{"text": "a"}]}) is None
    assert bench_serving._extract_stream_error(None) is None


def test_error_wins_over_output_in_same_payload():
    payload = {"choices": [{"text": "a"}], "error": {"message": "boom"}}
    has_output, stream_error = bench_serving._classify_stream_payload(payload)
    assert has_output is False
    assert stream_error is not None
    assert "boom" in stream_error


def test_tokens_then_error_is_failed():
    output = _Output()
    bench_serving._finalize_stream_output(
        output, stream_error=json.dumps(_error_payload()), first_chunk_received=True
    )
    assert not output.success
    assert _ERROR_MESSAGE in output.error


def test_clean_stream_is_successful():
    output = _Output()
    bench_serving._finalize_stream_output(
        output, stream_error=None, first_chunk_received=True
    )
    assert output.success
    assert output.error == ""


def test_no_chunk_without_error_reports_ttft_symptom():
    output = _Output()
    bench_serving._finalize_stream_output(
        output, stream_error=None, first_chunk_received=False
    )
    assert not output.success
    assert "Never received a valid chunk" in output.error


def test_error_takes_precedence_over_ttft_symptom():
    output = _Output()
    bench_serving._finalize_stream_output(
        output, stream_error=_ERROR_MESSAGE, first_chunk_received=False
    )
    assert not output.success
    assert output.error == _ERROR_MESSAGE


def test_manifest_rejects_top_level_error(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "hi", "error": _error_payload()}) + "\n"
    )
    with pytest.raises(ValueError, match="request contains a stream error"):
        request_manifest.load_request_manifest(manifest)


def test_manifest_rejects_sampling_params_error(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"prompt": "hi", "sampling_params": {"error": _ERROR_MESSAGE}}
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="sampling_params contains a stream error"):
        request_manifest.load_request_manifest(manifest)


def test_manifest_accepts_clean_request(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"prompt": "hi"}) + "\n")
    loaded = request_manifest.load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].prompt == "hi"
