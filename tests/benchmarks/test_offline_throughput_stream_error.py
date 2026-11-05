"""Lightweight tests for offline throughput stream error handling.

These tests load ``bench_offline_throughput`` by path so they do not require
the full sglang dependency set (torch, server args, etc.) to be installed.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
    / "bench_offline_throughput.py"
)


class _FakeResult:
    def __init__(self, output_file_path=None, error=None):
        self.output_file_path = output_file_path
        self.error = error


class _FakeEngine:
    def __init__(self, results):
        self._results = results

    def generate(self, sampling_params_kwargs=None):
        return self._results


def _load_bench_module():
    """Load the benchmark module with minimal stubs for heavy imports."""
    torch_stub = types.ModuleType("torch")
    torch_stub.get_device_module = lambda: types.SimpleNamespace(
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0,
    )
    sys.modules.setdefault("torch", torch_stub)

    spec = importlib.util.spec_from_file_location(
        "bench_offline_throughput_under_test", BENCH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench_module():
    return _load_bench_module()


def test_result_with_error_is_not_counted_as_success(bench_module, tmp_path):
    output_path = tmp_path / "out.png"
    output_path.write_bytes(b"partial")
    engine = _FakeEngine([_FakeResult(output_file_path=str(output_path), error="boom")])

    batch = bench_module.generate_batch(
        engine,
        bench_module.BenchArgs(),
        ["hello"],
        [{}],
        request_ids=["req-1"],
    )

    assert batch.num_samples == 0
    assert batch.success is False
    assert "boom" in batch.error
    assert len(batch.requests) == 1
    assert batch.requests[0].success is False
    assert batch.requests[0].output_file_paths == [str(output_path)]


def test_result_without_error_is_counted_as_success(bench_module, tmp_path):
    output_path = tmp_path / "out.png"
    output_path.write_bytes(b"ok")
    engine = _FakeEngine([_FakeResult(output_file_path=str(output_path))])

    batch = bench_module.generate_batch(
        engine,
        bench_module.BenchArgs(),
        ["hello"],
        [{}],
        request_ids=["req-1"],
    )

    assert batch.num_samples == 1
    assert batch.success is True
    assert batch.requests[0].success is True
    assert batch.requests[0].output_file_paths == [str(output_path)]


def test_empty_error_string_is_not_treated_as_failure(bench_module, tmp_path):
    output_path = tmp_path / "out.png"
    output_path.write_bytes(b"ok")
    engine = _FakeEngine([_FakeResult(output_file_path=str(output_path), error="  ")])

    batch = bench_module.generate_batch(
        engine,
        bench_module.BenchArgs(),
        ["hello"],
        [{}],
        request_ids=["req-1"],
    )

    assert batch.num_samples == 1
    assert batch.success is True
