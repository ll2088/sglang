"""Benchmark result parsing must fail on stream error envelopes."""

import json

import pytest

from results import BenchmarkResult


def _write(tmp_path, payload):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    return path


def test_top_level_error_is_rejected(tmp_path):
    path = _write(tmp_path, {"error": {"message": "boom", "code": 500}})
    with pytest.raises(ValueError, match="stream error"):
        BenchmarkResult.from_json(path)


def test_aggregated_error_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        {"aggregated_metrics": {"error": {"message": "boom"}}},
    )
    with pytest.raises(ValueError, match="stream error"):
        BenchmarkResult.from_json(path)


def test_stats_error_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        {"aggregated_metrics": {"stats": {"error": "boom"}}},
    )
    with pytest.raises(ValueError, match="stream error"):
        BenchmarkResult.from_json(path)


def test_clean_result_is_parsed(tmp_path):
    path = _write(
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
