"""Summary tests: streams ending in an error event must not count as success."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_PATH = (
    _REPO_ROOT / "sgl-model-gateway" / "e2e_test" / "benchmarks" / "results.py"
)
_SUMMARIZE_PATH = (
    _REPO_ROOT / "sgl-model-gateway" / "e2e_test" / "benchmarks" / "summarize.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results_module():
    return _load_module("bench_results_under_test", _RESULTS_PATH)


@pytest.fixture(scope="module")
def summarize_module():
    return _load_module("bench_summarize_under_test", _SUMMARIZE_PATH)


def _write_result(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_results_rejects_top_level_error(results_module, tmp_path):
    result = tmp_path / "result.json"
    _write_result(
        result,
        {
            "error": {"message": "boom", "code": 500},
            "aggregated_metrics": {"stats": {}},
        },
    )
    with pytest.raises(ValueError, match="stream error"):
        results_module.BenchmarkResult.from_json(result)


def test_results_rejects_aggregated_error(results_module, tmp_path):
    result = tmp_path / "result.json"
    _write_result(
        result,
        {
            "aggregated_metrics": {
                "error": {"message": "boom"},
                "stats": {},
            }
        },
    )
    with pytest.raises(ValueError, match="stream error"):
        results_module.BenchmarkResult.from_json(result)


def test_results_accepts_clean_payload(results_module, tmp_path):
    result = tmp_path / "result.json"
    _write_result(
        result,
        {
            "aggregated_metrics": {
                "stats": {
                    "ttft": {"mean": 0.1},
                    "e2e_latency": {"mean": 1.0},
                    "input_throughput": {"mean": 10.0},
                    "output_throughput": {"mean": 20.0},
                }
            }
        },
    )
    parsed = results_module.BenchmarkResult.from_json(result)
    assert parsed.ttft_mean == pytest.approx(0.1)
    assert parsed.output_throughput_mean == pytest.approx(20.0)


def test_summary_marks_error_stream_failed(summarize_module, tmp_path):
    folder = tmp_path / "benchmark_error_case"
    folder.mkdir()
    _write_result(
        folder / "result.json",
        {
            "error": {"message": "internal server error during generation"},
            "aggregated_metrics": {"stats": {}},
        },
    )
    summary = summarize_module.generate_summary(tmp_path)
    assert "error case" in summary
    assert "❌ Failed" in summary
    assert "✅ Success" not in summary


def test_summary_marks_clean_stream_success(summarize_module, tmp_path):
    folder = tmp_path / "benchmark_clean_case"
    folder.mkdir()
    _write_result(
        folder / "result.json",
        {
            "aggregated_metrics": {
                "stats": {
                    "ttft": {"mean": 0.1},
                    "e2e_latency": {"mean": 1.0},
                    "input_throughput": {"mean": 10.0},
                    "output_throughput": {"mean": 20.0},
                }
            }
        },
    )
    summary = summarize_module.generate_summary(tmp_path)
    assert "clean case" in summary
    assert "✅ Success" in summary
    assert "❌ Failed" not in summary
