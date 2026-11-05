"""Lightweight tests for offline benchmark stream error handling.

These tests load the benchmark modules by path so they do not require the
sglang package to be installed and do not collide with other test modules.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "python" / "sglang" / "multimodal_gen" / "benchmarks"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench_serving_module():
    return _load_module("bench_serving_under_test", BENCH_DIR / "bench_serving.py")


def test_stream_error_from_chunk_detects_error_envelope(bench_serving_module):
    chunk = (
        b'data: {"choices":[{"text":"a"}]}\n\n'
        b'data: {"error":{"message":"boom","code":500}}\n\n'
        b"data: [DONE]\n\n"
    )
    error = bench_serving_module._stream_error_from_chunk(chunk)
    assert error is not None
    assert "boom" in error


def test_stream_error_from_chunk_clean_stream(bench_serving_module):
    chunk = (
        b'data: {"choices":[{"text":"a"}]}\n\n'
        b'data: {"choices":[{"text":"b"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    assert bench_serving_module._stream_error_from_chunk(chunk) is None


def test_finalize_stream_output_error_wins_over_tokens(bench_serving_module):
    output = bench_serving_module.RequestFuncOutput()
    bench_serving_module._finalize_stream_output(
        output, stream_error='{"message": "boom"}', first_chunk_received=True
    )
    assert not output.success
    assert "boom" in output.error


def test_finalize_stream_output_clean_stream_succeeds(bench_serving_module):
    output = bench_serving_module.RequestFuncOutput()
    bench_serving_module._finalize_stream_output(
        output, stream_error=None, first_chunk_received=True
    )
    assert output.success


def test_finalize_stream_output_no_chunk_fails(bench_serving_module):
    output = bench_serving_module.RequestFuncOutput()
    bench_serving_module._finalize_stream_output(
        output, stream_error=None, first_chunk_received=False
    )
    assert not output.success
    assert "Never received a valid chunk" in output.error
