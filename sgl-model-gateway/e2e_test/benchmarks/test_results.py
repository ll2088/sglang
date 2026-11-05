"""Tests for benchmark result parsing and stream error handling."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_results_module():
    """Load results.py by path so the test does not need the package installed."""
    module_path = Path(__file__).with_name("results.py")
    spec = importlib.util.spec_from_file_location("bench_results", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BenchmarkResult = _load_results_module().BenchmarkResult


def _write_result(tmp_path: Path, payload) -> Path:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    return path


def test_result_with_stream_error_is_rejected(tmp_path):
    path = _write_result(
        tmp_path,
        {
            "error": {"message": "boom", "code": 500},
            "aggregated_metrics": {"stats": {}},
        },
    )
    with pytest.raises(ValueError, match="Benchmark stream error"):
        BenchmarkResult.from_json(path)


def test_clean_result_is_parsed(tmp_path):
    path = _write_result(
        tmp_path,
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
    result = BenchmarkResult.from_json(path)
    assert result.ttft_mean == pytest.approx(0.1)
    assert result.output_throughput_mean == pytest.approx(20.0)


def test_result_with_string_stream_error_is_rejected(tmp_path):
    path = _write_result(
        tmp_path,
        {
            "error": "internal server error during generation",
            "aggregated_metrics": {"stats": {}},
        },
    )
    with pytest.raises(ValueError, match="Benchmark stream error"):
        BenchmarkResult.from_json(path)


def test_result_with_null_error_is_parsed(tmp_path):
    path = _write_result(
        tmp_path,
        {
            "error": None,
            "aggregated_metrics": {
                "stats": {
                    "ttft": {"mean": 0.2},
                    "e2e_latency": {"mean": 2.0},
                    "input_throughput": {"mean": 5.0},
                    "output_throughput": {"mean": 7.0},
                }
            },
        },
    )
    result = BenchmarkResult.from_json(path)
    assert result.ttft_mean == pytest.approx(0.2)
    assert result.output_throughput_mean == pytest.approx(7.0)
