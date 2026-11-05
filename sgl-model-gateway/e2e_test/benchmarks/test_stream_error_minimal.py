"""Minimal-dependency tests for stream error handling in benchmark results.

These tests load ``results.py`` and ``request_manifest.py`` by path so they do
not import the sglang runtime. That keeps the stream-error assertions runnable
in a CPU-only environment where the full sglang dependency set is unavailable.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_BENCH_DIR = Path(__file__).resolve().parent


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _BENCH_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


results = _load_module("bench_results_under_test", "results.py")
request_manifest = _load_module("bench_request_manifest_under_test", "request_manifest.py")


def _write_result(tmp_path: Path, payload) -> Path:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload))
    return path


def test_stream_error_after_tokens_is_rejected(tmp_path):
    """A stream that delivered tokens then an error envelope must fail."""
    path = _write_result(
        tmp_path,
        {
            "error": {"message": "boom", "code": 500},
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
    with pytest.raises(ValueError, match="Benchmark stream error"):
        results.BenchmarkResult.from_json(path)


def test_clean_stream_is_parsed_as_success(tmp_path):
    """Guard against over-eager detection: a clean stream stays successful."""
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
    result = results.BenchmarkResult.from_json(path)
    assert result.ttft_mean == pytest.approx(0.1)
    assert result.output_throughput_mean == pytest.approx(20.0)


def test_manifest_line_with_stream_error_is_rejected(tmp_path):
    """A manifest record carrying an error envelope must not be accepted."""
    manifest = tmp_path / "workload.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "prompt": "hello",
                "error": {"message": "boom", "code": 500},
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="stream error"):
        request_manifest.load_request_manifest(manifest)


def test_manifest_clean_line_is_loaded(tmp_path):
    """A clean manifest line still loads, so detection is not over-eager."""
    manifest = tmp_path / "workload.jsonl"
    manifest.write_text(json.dumps({"prompt": "hello"}) + "\n")
    loaded = request_manifest.load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].prompt == "hello"
