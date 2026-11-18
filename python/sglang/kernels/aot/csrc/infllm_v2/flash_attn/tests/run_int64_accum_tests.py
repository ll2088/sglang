"""Standalone runner for the int64 accumulator tests.

This entry point is intentionally stdlib-only and does not import ``sglang``,
``torch``, ``numpy`` or ``pytest``. It exists so that CI can execute the int64
overflow assertions even when the full sglang dependency set cannot be
installed (e.g. ``pip install -e python`` fails during environment
preparation). It simply imports the sibling test module and runs every
``test_*`` function defined there.

Usage::

    python run_int64_accum_tests.py

Exit code is 0 when all tests pass, 1 otherwise.
"""

import os
import sys


def _load_test_module():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import test_int64_accum_sizes  # noqa: WPS433 (local import by design)

    return test_int64_accum_sizes


def main():
    module = _load_test_module()
    tests = module._collect_tests()
    failures = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures.append((name, exc))
    if failures:
        for name, exc in failures:
            print("FAILED {}: {}".format(name, exc))
        return 1
    print("All {} int64 accumulator tests passed.".format(len(tests)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
