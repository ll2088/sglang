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


def test_result_with_nested_aggregated_error_is_rejected(tmp_path):
    path = _write_result(
        tmp_path,
        {
            "aggregated_metrics": {
                "error": {"message": "boom", "code": 500},
                "stats": {},
            }
        },
    )
    with pytest.raises(ValueError, match="Benchmark stream error"):
        BenchmarkResult.from_json(path)


def test_result_with_stats_error_is_rejected(tmp_path):
    path = _write_result(
        tmp_path,
        {
            "aggregated_metrics": {
                "stats": {"error": "internal server error during generation"}
            }
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


def test_summarize_excludes_stream_error_from_success(tmp_path):
    """A result carrying a stream error must not be counted as successful."""
    summarize_path = Path(__file__).with_name("summarize.py")
    spec = importlib.util.spec_from_file_location("bench_summarize", summarize_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    folder = tmp_path / "benchmark_stream_error"
    folder.mkdir()
    (folder / "result.json").write_text(
        json.dumps(
            {
                "error": {"message": "boom", "code": 500},
                "aggregated_metrics": {"stats": {}},
            }
        )
    )

    summary = module.generate_summary(tmp_path)
    assert "❌ Failed" in summary
    assert "✅ Success" not in summary


def test_summarize_counts_clean_result_as_success(tmp_path):
    """A clean result must still be reported as successful."""
    summarize_path = Path(__file__).with_name("summarize.py")
    spec = importlib.util.spec_from_file_location("bench_summarize_clean", summarize_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    folder = tmp_path / "benchmark_clean"
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


def test_result_with_error_after_partial_metrics_is_rejected(tmp_path):
    """A stream that delivered tokens then failed must not count as success.

    The server reports a mid-stream failure as an ``error`` envelope after
    partial metrics were already aggregated. The error is the root cause and
    must win over the partial output, which is only a symptom.
    """
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
            },
            "error": {"message": "internal server error during generation"},
        },
    )
    with pytest.raises(ValueError, match="Benchmark stream error"):
        BenchmarkResult.from_json(path)


def test_summarize_excludes_error_after_partial_metrics(tmp_path):
    """A result with partial metrics and a stream error is not successful."""
    summarize_path = Path(__file__).with_name("summarize.py")
    spec = importlib.util.spec_from_file_location(
        "bench_summarize_partial_error", summarize_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    folder = tmp_path / "benchmark_partial_error"
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
                },
                "error": "internal server error during generation",
            }
        )
    )

    summary = module.generate_summary(tmp_path)
    assert "❌ Failed" in summary
    assert "✅ Success" not in summary
