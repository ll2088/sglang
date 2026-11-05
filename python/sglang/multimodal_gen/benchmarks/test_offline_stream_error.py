"""Tests that offline throughput treats result errors as failures."""

import importlib.util
import sys
from pathlib import Path


def _load_bench_offline():
    module_path = Path(__file__).with_name("bench_offline_throughput.py")
    spec = importlib.util.spec_from_file_location("bench_offline_throughput", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench_offline = _load_bench_offline()


class _Result:
    def __init__(self, error=None, output_file_path=None):
        self.error = error
        self.output_file_path = output_file_path


class _Engine:
    def __init__(self, results):
        self._results = results

    def generate(self, sampling_params_kwargs):
        return self._results


def test_generate_batch_marks_error_result_as_failed(tmp_path):
    engine = _Engine([_Result(error="boom", output_file_path=None)])
    bench_args = bench_offline.BenchArgs()
    output = bench_offline.generate_batch(
        engine, bench_args, ["prompt"], [{"num_frames": 1}], request_ids=["r1"]
    )
    assert output.num_samples == 0
    assert output.requests[0].success is False
    assert "boom" in output.requests[0].error


def test_generate_batch_marks_clean_result_as_success(tmp_path):
    output_path = tmp_path / "out.png"
    output_path.write_bytes(b"data")
    engine = _Engine([_Result(error=None, output_file_path=str(output_path))])
    bench_args = bench_offline.BenchArgs()
    output = bench_offline.generate_batch(
        engine, bench_args, ["prompt"], [{"num_frames": 1}], request_ids=["r1"]
    )
    assert output.num_samples == 1
    assert output.requests[0].success is True
