"""Stream error events must fail benchmark requests, not count as success.

A generation that fails after the response has started cannot change the
HTTP 200 already sent, so the server reports the failure as an ``error``
envelope inside the stream. These tests pin the benchmark client rule: a
stream that delivered tokens and then failed is not successful, while a
clean stream still is.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCH_SERVING = (
    _REPO_ROOT
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
    / "bench_serving.py"
)
_REQUEST_MANIFEST = (
    _REPO_ROOT
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
    / "request_manifest.py"
)
_RESULTS = _REPO_ROOT / "sgl-model-gateway" / "e2e_test" / "benchmarks" / "results.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench_serving = _load_module("bench_serving_under_test", _BENCH_SERVING)
request_manifest = _load_module("request_manifest_under_test", _REQUEST_MANIFEST)
results = _load_module("bench_results_under_test", _RESULTS)


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


def _token_payload(text: str) -> dict:
    return {"choices": [{"text": text}]}


def _sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def test_stream_error_wins_over_partial_tokens():
    """Tokens followed by an error envelope must be classified as failed."""
    chunks = [
        _sse(_token_payload("a")),
        _sse(_token_payload("b")),
        _sse(_error_payload()),
        b"data: [DONE]\n\n",
    ]

    saw_output = False
    stream_error = None
    for chunk in chunks:
        for payload in bench_serving._iter_stream_payloads(chunk):
            has_output, error = bench_serving._classify_stream_payload(payload)
            if error is not None:
                stream_error = error
            elif has_output:
                saw_output = True

    assert saw_output, "tokens must have been observed before the error"
    assert stream_error is not None
    assert _ERROR_MESSAGE in stream_error

    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(output, stream_error, saw_output)
    assert not output.success
    assert _ERROR_MESSAGE in output.error


def test_clean_stream_is_still_successful():
    """A stream without an error envelope must remain successful."""
    chunks = [
        _sse(_token_payload("a")),
        _sse(_token_payload("b")),
        b"data: [DONE]\n\n",
    ]

    saw_output = False
    stream_error = None
    for chunk in chunks:
        for payload in bench_serving._iter_stream_payloads(chunk):
            has_output, error = bench_serving._classify_stream_payload(payload)
            if error is not None:
                stream_error = error
            elif has_output:
                saw_output = True

    assert saw_output
    assert stream_error is None

    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(output, stream_error, saw_output)
    assert output.success
    assert output.error == ""


def test_stream_error_takes_precedence_over_no_chunk_symptom():
    """The server error must not be replaced by the TTFT symptom message."""
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(output, _ERROR_MESSAGE, False)
    assert not output.success
    assert output.error == _ERROR_MESSAGE


def test_manifest_rejects_stream_error(tmp_path: Path):
    """A manifest line carrying a stream error must be rejected."""
    manifest = tmp_path / "workload.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "hello", "error": _ERROR_MESSAGE}) + "\n"
    )
    with pytest.raises(ValueError, match="stream error"):
        request_manifest.load_request_manifest(manifest)


def test_manifest_accepts_clean_line(tmp_path: Path):
    """A clean manifest line must still load."""
    manifest = tmp_path / "workload.jsonl"
    manifest.write_text(json.dumps({"prompt": "hello"}) + "\n")
    loaded = request_manifest.load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].prompt == "hello"


def test_benchmark_result_with_stream_error_is_rejected(tmp_path: Path):
    """A benchmark result carrying a stream error must not be parsed."""
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "error": {"message": _ERROR_MESSAGE, "code": 500},
                "aggregated_metrics": {"stats": {}},
            }
        )
    )
    with pytest.raises(ValueError, match="Benchmark stream error"):
        results.BenchmarkResult.from_json(path)


def test_benchmark_result_clean_is_parsed(tmp_path: Path):
    """A clean benchmark result must still be parsed."""
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "aggregated_metrics": {
                    "stats": {
                        "ttft": {"mean": 0.1},
                        "e2e_latency": {"mean": 1.0},
                        "input_throughput": {"mean": 10.0},
                        "output_throughput": {"mean": 20.0},
                    }
                }
            }
        )
    )
    result = results.BenchmarkResult.from_json(path)
    assert result.ttft_mean == pytest.approx(0.1)
    assert result.output_throughput_mean == pytest.approx(20.0)
