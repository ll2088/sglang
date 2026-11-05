"""Tests for benchmark result parsing and stream error handling."""

import json

import pytest

from results import BenchmarkResult


STREAM_ERROR_MESSAGE = "internal server error during generation"


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_benchmark_result_rejects_stream_error(tmp_path) -> None:
    path = _write(
        tmp_path / "result.json",
        {
            "error": {
                "message": STREAM_ERROR_MESSAGE,
                "type": "InternalServerError",
                "code": 500,
            },
            "aggregated_metrics": {"stats": {}},
        },
    )
    with pytest.raises(ValueError, match="stream error"):
        BenchmarkResult.from_json(path)


def test_benchmark_result_accepts_normal_payload(tmp_path) -> None:
    path = _write(
        tmp_path / "result.json",
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


def test_benchmark_result_rejects_string_stream_error(tmp_path) -> None:
    path = _write(
        tmp_path / "result.json",
        {"error": STREAM_ERROR_MESSAGE, "aggregated_metrics": {"stats": {}}},
    )
    with pytest.raises(ValueError, match="stream error"):
        BenchmarkResult.from_json(path)


def test_benchmark_result_accepts_null_error(tmp_path) -> None:
    path = _write(
        tmp_path / "result.json",
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


def test_benchmark_result_rejects_error_after_partial_metrics(tmp_path) -> None:
    """A stream that delivered tokens and then an error event must fail."""
    path = _write(
        tmp_path / "result.json",
        {
            "error": {
                "message": STREAM_ERROR_MESSAGE,
                "type": "InternalServerError",
                "code": 500,
            },
            "aggregated_metrics": {
                "stats": {
                    "ttft": {"mean": 0.1},
                    "e2e_latency": {"mean": 1.0},
                    "input_throughput": {"mean": 10.0},
                    "output_throughput": {"mean": 20.0},
                }
            },
        },
    )
    with pytest.raises(ValueError, match="stream error"):
        BenchmarkResult.from_json(path)
