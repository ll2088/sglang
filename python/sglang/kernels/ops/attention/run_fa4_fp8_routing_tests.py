# Copyright (c) 2026, SGLang Team.
"""Minimal runner for the standalone FA4 SM90 FP8 routing tests.

Usage (no sglang install required):
    python python/sglang/kernels/ops/attention/run_fa4_fp8_routing_tests.py
"""

import pathlib
import sys


def main() -> int:
    import pytest

    here = pathlib.Path(__file__).resolve().parent
    targets = [
        str(here / "test_fa4_sm90_fp8_policy_standalone.py"),
        str(here / "test_flash_attention_v4_fp8_routing.py"),
    ]
    return pytest.main(["-q", "-p", "no:randomly", *targets])


if __name__ == "__main__":
    sys.exit(main())
