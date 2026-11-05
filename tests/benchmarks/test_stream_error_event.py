"""Tests for stream error event handling in multimodal benchmark clients.

A stream that delivers tokens and then a server ``error`` envelope must be
reported as failed, with the server's error object preserved. Partial output
may be kept for diagnostics but must never count as success.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# This is the single authoritative test module for stream error event
# handling in benchmark clients. Do not add duplicate copies under
# python/.../benchmarks/ or test/benchmark/; pytest collection conflicts
# and CI instability result from duplicated module names.


_BENCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
)


def _load_module(name: str, filename: str):
    """Load a benchmark module by path without importing the sglang package."""
    module_path = Path(filename)
    if not module_path.is_absolute():
        module_path = _BENCH_DIR / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench_serving():
    return _load_module("bench_serving_under_test", "bench_serving.py")


@pytest.fixture(scope="module")
def request_manifest():
    return _load_module("request_manifest_under_test", "request_manifest.py")


_STREAM_ERROR_MESSAGE = "internal server error during generation"


def _error_payload(message: str = _STREAM_ERROR_MESSAGE) -> dict:
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


def test_extract_stream_error_returns_json_for_error_envelope(bench_serving):
    payload = _error_payload()
    extracted = bench_serving._extract_stream_error(payload)
    assert extracted is not None
    assert _STREAM_ERROR_MESSAGE in extracted


def test_extract_stream_error_returns_none_for_token_payload(bench_serving):
    assert bench_serving._extract_stream_error(_token_payload("a")) is None


def test_classify_stream_payload_error_wins_over_output(bench_serving):
    payload = _token_payload("a")
    payload.update(_error_payload())
    has_output, stream_error = bench_serving._classify_stream_payload(payload)
    assert has_output is False
    assert stream_error is not None
    assert _STREAM_ERROR_MESSAGE in stream_error


def test_finalize_stream_output_marks_error_as_failure(bench_serving):
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(
        output, json.dumps(_error_payload()), first_chunk_received=True
    )
    assert output.success is False
    assert _STREAM_ERROR_MESSAGE in output.error


def test_finalize_stream_output_clean_stream_is_success(bench_serving):
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(output, None, first_chunk_received=True)
    assert output.success is True
    assert output.error == ""


def test_finalize_stream_output_no_chunk_reports_symptom(bench_serving):
    output = bench_serving.RequestFuncOutput()
    bench_serving._finalize_stream_output(output, None, first_chunk_received=False)
    assert output.success is False
    assert "Never received a valid chunk" in output.error


def test_request_manifest_rejects_top_level_error(request_manifest, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "hello", "error": _error_payload()}) + "\n"
    )
    with pytest.raises(ValueError, match="stream error"):
        request_manifest.load_request_manifest(manifest)


def test_request_manifest_rejects_sampling_params_error(request_manifest, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"prompt": "hello", "sampling_params": {"error": _error_payload()}}
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="stream error"):
        request_manifest.load_request_manifest(manifest)


def test_request_manifest_accepts_clean_request(request_manifest, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"prompt": "hello"}) + "\n")
    loaded = request_manifest.load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].prompt == "hello"


def test_benchmark_result_rejects_stream_error(tmp_path):
    """A benchmark result carrying a stream error must not be a success."""
    results_path = (
        Path(__file__).resolve().parents[2]
        / "sgl-model-gateway"
        / "e2e_test"
        / "benchmarks"
        / "results.py"
    )
    module = _load_module("benchmark_results_under_test", str(results_path))
    result_file = tmp_path / "result.json"
    result_file.write_text(
        json.dumps(
            {
                "error": {"message": _STREAM_ERROR_MESSAGE, "code": 500},
                "aggregated_metrics": {"stats": {}},
            }
        )
    )
    with pytest.raises(ValueError, match="stream error"):
        module.BenchmarkResult.from_json(result_file)


def test_benchmark_result_accepts_clean_stream(tmp_path):
    """A clean benchmark result must still parse as successful."""
    results_path = (
        Path(__file__).resolve().parents[2]
        / "sgl-model-gateway"
        / "e2e_test"
        / "benchmarks"
        / "results.py"
    )
    module = _load_module("benchmark_results_under_test", str(results_path))
    result_file = tmp_path / "result.json"
    result_file.write_text(
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
    result = module.BenchmarkResult.from_json(result_file)
    assert result.ttft_mean == pytest.approx(0.1)
    assert result.output_throughput_mean == pytest.approx(20.0)


def test_benchmark_result_rejects_nested_stats_error(tmp_path):
    """An error nested in aggregated_metrics.stats must also fail the run."""
    results_path = (
        Path(__file__).resolve().parents[2]
        / "sgl-model-gateway"
        / "e2e_test"
        / "benchmarks"
        / "results.py"
    )
    module = _load_module("benchmark_results_under_test", str(results_path))
    result_file = tmp_path / "result.json"
    result_file.write_text(
        json.dumps(
            {
                "aggregated_metrics": {
                    "stats": {"error": {"message": _STREAM_ERROR_MESSAGE}}
                }
            }
        )
    )
    with pytest.raises(ValueError, match="stream error"):
        module.BenchmarkResult.from_json(result_file)


def test_summarize_marks_error_result_as_failed(tmp_path):
    """summarize must not count an error-terminated stream as success."""
    summarize_path = (
        Path(__file__).resolve().parents[2]
        / "sgl-model-gateway"
        / "e2e_test"
        / "benchmarks"
        / "summarize.py"
    )
    module = _load_module("benchmark_summarize_under_test", str(summarize_path))
    folder = tmp_path / "benchmark_error_case"
    folder.mkdir()
    (folder / "result.json").write_text(
        json.dumps(
            {
                "error": {"message": _STREAM_ERROR_MESSAGE, "code": 500},
                "aggregated_metrics": {"stats": {}},
            }
        )
    )
    summary = module.generate_summary(tmp_path)
    assert "❌ Failed" in summary
    assert "✅ Success" not in summary


def test_summarize_marks_clean_result_as_success(tmp_path):
    """A clean result must still be reported as success by summarize."""
    summarize_path = (
        Path(__file__).resolve().parents[2]
        / "sgl-model-gateway"
        / "e2e_test"
        / "benchmarks"
        / "summarize.py"
    )
    module = _load_module("benchmark_summarize_under_test", str(summarize_path))
    folder = tmp_path / "benchmark_clean_case"
    folder.mkdir()
    (folder / "result.json").write_text(
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
    summary = module.generate_summary(tmp_path)
    assert "✅ Success" in summary
    assert "❌ Failed" not in summary
