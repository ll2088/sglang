"""CPU-only tests for gateway benchmark result stream error handling."""

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "sgl-model-gateway" / "e2e_test" / "benchmarks" / "results.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results_module():
    return _load_module("gateway_results_under_test", RESULTS)


def _write(tmp_path: Path, data) -> Path:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data))
    return path


def test_top_level_error_is_rejected(results_module, tmp_path):
    path = _write(tmp_path, {"error": {"message": "boom"}})
    with pytest.raises(ValueError, match="stream error"):
        results_module.BenchmarkResult.from_json(path)


def test_aggregated_error_is_rejected(results_module, tmp_path):
    path = _write(
        tmp_path,
        {"aggregated_metrics": {"error": {"message": "boom"}}},
    )
    with pytest.raises(ValueError, match="stream error"):
        results_module.BenchmarkResult.from_json(path)


def test_stats_error_is_rejected(results_module, tmp_path):
    path = _write(
        tmp_path,
        {"aggregated_metrics": {"stats": {"error": "boom"}}},
    )
    with pytest.raises(ValueError, match="stream error"):
        results_module.BenchmarkResult.from_json(path)


def test_clean_result_is_parsed(results_module, tmp_path):
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
    result = results_module.BenchmarkResult.from_json(path)
    assert result.ttft_mean == pytest.approx(0.1)
    assert result.output_throughput_mean == pytest.approx(20.0)
