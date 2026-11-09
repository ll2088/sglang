# Copyright (c) 2026, SGLang Team.
"""Minimal-dependency entry point for the FA4 SM90 FP8 routing tests.

CI can run this without `pip install -e python`: it only needs pytest and
CPU torch, and it loads the pure policy module by path.

Usage:
    python python/sglang/kernels/ops/attention/run_fa4_fp8_routing_tests.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
TESTS = [
    HERE / "test_flash_attention_v4_fp8_routing.py",
    HERE / "test_fa4_sm90_fp8_mma_numerics.py",
    HERE / "test_bench_flash_attention_fp8_output.py",
]


def main() -> int:
    import pytest

    args = [str(p) for p in TESTS if p.is_file()]
    if not args:
        print("no routing tests found", file=sys.stderr)
        return 1
    return pytest.main(["-q", "-p", "no:randomly", *args])


if __name__ == "__main__":
    raise SystemExit(main())
