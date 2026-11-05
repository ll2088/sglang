"""CPU-only tests for offline throughput stream error handling."""

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFLINE = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
    / "bench_offline_throughput.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def offline():
    pytest.importorskip("torch")
    pytest.importorskip("tqdm")
    return _load_module("bench_offline_under_test", OFFLINE)


class _Result:
    def __init__(self, error=None, output_file_path=None):
        self.error = error
        self.output_file_path = output_file_path


def test_first_result_error_detects_error(offline):
    results = [_Result(output_file_path="a.png"), _Result(error="boom")]
    assert offline._first_result_error(results) == "boom"


def test_first_result_error_absent(offline):
    results = [_Result(output_file_path="a.png")]
    assert offline._first_result_error(results) is None


def test_result_has_output(offline):
    assert offline._result_has_output(_Result(output_file_path="a.png")) is True
    assert offline._result_has_output(_Result()) is False
