"""Offline throughput benchmark must not count error results as success."""

import importlib.util
import sys
from pathlib import Path

import pytest


_BENCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
)


def _load_module(name: str, filename: str):
    module_path = _BENCH_DIR / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench_offline = _load_module(
    "bench_offline_throughput_stream_error", "bench_offline_throughput.py"
)


class _Result:
    def __init__(self, output_file_path=None, error=None):
        self.output_file_path = output_file_path
        self.error = error


class _Engine:
    def __init__(self, results):
        self._results = results

    def generate(self, sampling_params_kwargs):
        return self._results


class _TorchDevice:
    def reset_peak_memory_stats(self):
        pass

    def max_memory_allocated(self):
        return 0


class _Torch:
    def get_device_module(self):
        return _TorchDevice()


@pytest.fixture(autouse=True)
def _patch_torch(monkeypatch):
    monkeypatch.setattr(bench_offline, "torch", _Torch())


def _run(engine):
    return bench_offline.generate_batch(
        engine,
        bench_offline.BenchArgs(),
        ["prompt"],
        [{"num_frames": 1}],
        request_ids=["request-00000"],
    )


def test_result_with_error_is_failed():
    output = _run(_Engine([_Result(output_file_path=None, error="boom")]))
    assert output.num_samples == 0
    assert output.requests[0].success is False
    assert "boom" in output.requests[0].error


def test_result_without_output_is_failed():
    output = _run(_Engine([_Result(output_file_path=None, error=None)]))
    assert output.num_samples == 0
    assert output.requests[0].success is False


def test_clean_result_is_successful(tmp_path: Path):
    generated = tmp_path / "out.png"
    generated.write_bytes(b"data")
    output = _run(_Engine([_Result(output_file_path=str(generated), error=None)]))
    assert output.num_samples == 1
    assert output.requests[0].success is True
