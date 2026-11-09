# Copyright (c) 2026, SGLang Team.
"""Output-format check for the FA4 FP8 benchmark's FP8-vs-BF16 summary.

Runs on CPU: it only exercises the pure formatting helper, so the benchmark
script can be validated without a GPU or a compiled extension.
"""

import importlib.util
import pathlib
import sys

_BENCH_PATH = (
    pathlib.Path(__file__).resolve().parents[5]
    / "benchmark"
    / "kernels"
    / "attention"
    / "bench_flash_attention_fp8.py"
)


def _load_bench_module():
    spec = importlib.util.spec_from_file_location(
        "_bench_flash_attention_fp8_under_test", _BENCH_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover
        sys.modules.pop(spec.name, None)
        raise
    return module


def test_bench_file_exists():
    assert _BENCH_PATH.is_file(), f"missing benchmark script: {_BENCH_PATH}"


def test_fp8_vs_bf16_summary_reports_median_and_speedup():
    bench = _load_bench_module()
    line = bench.format_fp8_vs_bf16_summary(
        fp8_median_ms=1.25, bf16_median_ms=1.50
    )
    assert "FP8" in line
    assert "BF16" in line
    assert "median" in line.lower()
    assert "1.25" in line
    assert "1.50" in line
    assert "1.20x" in line


def test_fp8_vs_bf16_summary_flags_regression():
    bench = _load_bench_module()
    line = bench.format_fp8_vs_bf16_summary(
        fp8_median_ms=2.00, bf16_median_ms=1.50
    )
    assert "WARNING" in line
