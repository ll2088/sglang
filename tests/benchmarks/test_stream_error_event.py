"""Lightweight tests for stream error detection in benchmark clients.

These tests only import standard-library modules and the benchmark helper
modules by path, so they run without installing the full sglang package.
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
BENCH_OFFLINE = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
    / "bench_offline_throughput.py"
)
MANIFEST = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
    / "request_manifest.py"
)
RESULTS = REPO_ROOT / "sgl-model-gateway" / "e2e_test" / "benchmarks" / "results.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sse(payload) -> bytes:
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


def _token_chunk(text: str = "tok") -> bytes:
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


@pytest.fixture(scope="module")
def bench_serving_module():
    return _load_module("bench_serving_under_test", BENCH_SERVING)


@pytest.fixture(scope="module")
def bench_offline_module():
    return _load_module("bench_offline_under_test", BENCH_OFFLINE)


@pytest.fixture(scope="module")
def manifest_module():
    return _load_module("request_manifest_under_test", MANIFEST)


@pytest.fixture(scope="module")
def results_module():
    return _load_module("results_under_test", RESULTS)


def test_stream_error_detected_after_tokens(bench_serving_module):
    body = _token_chunk() + _token_chunk() + _error_chunk() + b"data: [DONE]\n\n"
    error = bench_serving_module._stream_error_from_chunk(body)
    assert error is not None
    assert "internal server error" in error


def test_stream_error_absent_for_clean_stream(bench_serving_module):
    body = _token_chunk() + _token_chunk() + b"data: [DONE]\n\n"
    assert bench_serving_module._stream_error_from_chunk(body) is None


def test_stream_error_wins_over_output_in_same_chunk(bench_serving_module):
    payload = {
        "choices": [{"text": "partial", "index": 0}],
        "error": {"message": "boom", "code": 500},
    }
    has_output, error = bench_serving_module._classify_stream_payload(payload)
    assert has_output is False
    assert error is not None and "boom" in error


def test_finalize_stream_output_marks_error_as_failure(bench_serving_module):
    output = bench_serving_module.RequestFuncOutput()
    bench_serving_module._finalize_stream_output(output, "boom", True)
    assert output.success is False
    assert output.error == "boom"


def test_finalize_stream_output_marks_clean_stream_as_success(bench_serving_module):
    output = bench_serving_module.RequestFuncOutput()
    bench_serving_module._finalize_stream_output(output, None, True)
    assert output.success is True


def test_finalize_stream_output_marks_empty_stream_as_failure(bench_serving_module):
    output = bench_serving_module.RequestFuncOutput()
    bench_serving_module._finalize_stream_output(output, None, False)
    assert output.success is False
    assert "Never received a valid chunk" in output.error


def test_offline_result_error_is_detected(bench_offline_module):
    class Result:
        error = "generation failed"
        output_file_path = "/tmp/partial.png"

    assert bench_offline_module._first_result_error([Result()]) == "generation failed"


def test_offline_clean_result_has_no_error(bench_offline_module):
    class Result:
        error = ""
        output_file_path = "/tmp/out.png"

    assert bench_offline_module._first_result_error([Result()]) is None


def _write_manifest(tmp_path: Path, records) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_manifest_rejects_stream_error(manifest_module, tmp_path):
    path = _write_manifest(
        tmp_path,
        [{"prompt": "hello", "error": {"message": "boom", "code": 500}}],
    )
    with pytest.raises(ValueError, match="stream error"):
        manifest_module.load_request_manifest(path)


def test_manifest_accepts_clean_request(manifest_module, tmp_path):
    path = _write_manifest(tmp_path, [{"prompt": "hello"}])
    loaded = manifest_module.load_request_manifest(path)
    assert len(loaded.requests) == 1


def test_results_rejects_stream_error(results_module, tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"error": {"message": "boom"}}))
    with pytest.raises(ValueError, match="stream error"):
        results_module.BenchmarkResult.from_json(path)


def test_results_accepts_clean_payload(results_module, tmp_path):
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "aggregated_metrics": {
                    "stats": {
                        "ttft": {"mean": 0.1},
                        "e2e_latency": {"mean": 0.2},
                        "input_throughput": {"mean": 10.0},
                        "output_throughput": {"mean": 20.0},
                    }
                }
            }
        )
    )
    result = results_module.BenchmarkResult.from_json(path)
    assert result.ttft_mean == pytest.approx(0.1)
