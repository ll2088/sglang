"""Lightweight tests for stream error detection in benchmark clients.

These tests avoid importing sglang so they can run in a minimal environment.
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


def _load_bench_serving():
    spec = importlib.util.spec_from_file_location("bench_serving_under_test", BENCH_SERVING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench_serving():
    return _load_bench_serving()


ERROR_MESSAGE = "internal server error during generation"


def _error_payload():
    return {
        "error": {
            "message": ERROR_MESSAGE,
            "type": "InternalServerError",
            "param": None,
            "code": 500,
        }
    }


def test_extract_stream_error_returns_serialized_error(bench_serving):
    error = bench_serving._extract_stream_error(_error_payload())
    assert error is not None
    assert ERROR_MESSAGE in error


def test_extract_stream_error_accepts_string(bench_serving):
    assert bench_serving._extract_stream_error({"error": "boom"}) == "boom"


def test_extract_stream_error_absent(bench_serving):
    assert bench_serving._extract_stream_error({"choices": [{"text": "a"}]}) is None
    assert bench_serving._extract_stream_error(None) is None


def test_error_envelope_wins_over_output(bench_serving):
    payload = {
        "choices": [{"text": "partial"}],
        "error": {"message": ERROR_MESSAGE, "code": 500},
    }
    error = bench_serving._extract_stream_error(payload)
    assert error is not None
    assert ERROR_MESSAGE in error


def test_clean_payload_has_no_error(bench_serving):
    payload = {"choices": [{"text": "a"}, {"text": "b"}]}
    assert bench_serving._extract_stream_error(payload) is None


def test_error_payload_is_json_serializable(bench_serving):
    error = bench_serving._extract_stream_error(_error_payload())
    assert json.loads(error)["message"] == ERROR_MESSAGE
