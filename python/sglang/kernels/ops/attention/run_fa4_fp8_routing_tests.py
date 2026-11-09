# Copyright (c) 2026, SGLang Team.
"""Torch-free entry point for the FA4 SM90 FP8 routing checks.

Usage: python python/sglang/kernels/ops/attention/run_fa4_fp8_routing_tests.py
"""

import pathlib
import sys
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from test_fa4_sm90_fp8_policy_standalone import (  # noqa: E402
    DescaleTest,
    RoutingTest,
    SplitTest,
)


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for case in (RoutingTest, SplitTest, DescaleTest):
        suite.addTests(loader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
