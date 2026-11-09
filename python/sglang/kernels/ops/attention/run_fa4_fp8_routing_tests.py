# Copyright (c) 2026, SGLang Team.
"""Run the FA4 SM90 FP8 routing checks without installing sglang.

Usage: python python/sglang/kernels/ops/attention/run_fa4_fp8_routing_tests.py
"""

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    targets = [
        str(HERE / "test_fa4_sm90_fp8_policy_standalone.py"),
        str(HERE / "test_flash_attention_v4_fp8_routing.py"),
    ]
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", *targets]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
