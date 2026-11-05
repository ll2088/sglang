"""Tests that benchmark summaries exclude streams ending in a server error."""

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "sgl-model-gateway" / "e2e_test" / "benchmarks" / "results.py"
SUMMARIZE = REPO_ROOT / "sgl-model-gateway" / "e2e_test" / "benchmarks" / "summarize.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results_module():
    return _load_module("benchmark_results_under_test", RESULTS)


def _write_result(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _clean_payload() -> dict:
    return {
        "aggregated_metrics": {
            "stats": {
                "ttft": {"mean": 0.1},
                "e2e_latency": {"mean": 1.0},
                "input_throughput": {"mean": 100.0},
                "output_throughput": {"mean": 50.0},
            }
        }
    }


def test_clean_result_is_parsed(results_module, tmp_path):
    path = _write_result(tmp_path / "clean.json", _clean_payload())
    result = results_module.BenchmarkResult.from_json(path)
    assert result.output_throughput_mean == 50.0


def test_top_level_error_is_rejected(results_module, tmp_path):
    payload = _clean_payload()
    payload["error"] = {"message": "boom", "code": 500}
    path = _write_result(tmp_path / "top.json", payload)
    with pytest.raises(ValueError, match="stream error"):
        results_module.BenchmarkResult.from_json(path)


def test_nested_error_is_rejected(results_module, tmp_path):
    payload = _clean_payload()
    payload["aggregated_metrics"]["stats"]["error"] = "internal server error"
    path = _write_result(tmp_path / "nested.json", payload)
    with pytest.raises(ValueError, match="stream error"):
        results_module.BenchmarkResult.from_json(path)


def test_summary_marks_error_stream_as_failed(tmp_path):
    summarize = _load_module("benchmark_summarize_under_test", SUMMARIZE)
    folder = tmp_path / "benchmark_error_case"
    folder.mkdir()
    payload = _clean_payload()
    payload["error"] = {"message": "boom", "code": 500}
    _write_result(folder / "result.json", payload)

    summary = summarize.generate_summary(tmp_path)
    assert "error case" in summary
    assert "Failed" in summary
    assert "Success" not in summary


def test_summary_keeps_clean_stream_as_success(tmp_path):
    summarize = _load_module("benchmark_summarize_clean_under_test", SUMMARIZE)
    folder = tmp_path / "benchmark_clean_case"
    folder.mkdir()
    _write_result(folder / "result.json", _clean_payload())

    summary = summarize.generate_summary(tmp_path)
    assert "clean case" in summary
    assert "Success" in summary
    assert "Failed" not in summary
