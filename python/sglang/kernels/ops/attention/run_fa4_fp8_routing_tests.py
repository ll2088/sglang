# Copyright (c) 2026, SGLang Team.
"""Standalone entry point for the FA4 SM90 FP8 routing tests.

Runs the routing/descale/split tests without installing the sglang package
and without initializing CUDA, so CI does not need `pip install -e python`.

Usage:
    python python/sglang/kernels/ops/attention/run_fa4_fp8_routing_tests.py
"""

from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    try:
        import pytest
    except ImportError:  # pragma: no cover - CI installs pytest
        print("pytest is required to run the FA4 FP8 routing tests", file=sys.stderr)
        return 2

    args = list(sys.argv[1:] if argv is None else argv)
    target = str(_HERE / "test_flash_attention_v4_fp8_routing.py")
    return pytest.main(["-q", "-p", "no:randomly", target, *args])


if __name__ == "__main__":
    raise SystemExit(main())
