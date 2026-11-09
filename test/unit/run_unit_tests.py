"""Minimal launcher for the run_suite unit tests.

These tests only exercise run_suite.py's exit-code and collection-diagnostic
branches, so they must be runnable with nothing but pytest installed. CI
verification jobs can call this script instead of installing the full sglang
dependency set (which is what previously failed during environment setup).

Usage (from the repository root):

    python test/unit/run_unit_tests.py

The script is intentionally dependency-free: it only imports the standard
library and pytest, and it pins the same rootdir/testpaths as
``test/unit/pytest.ini`` so local runs and CI agree.
"""

import os
import sys


UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # run_suite.py lives in test/ and imports sglang.test.ci.*; conftest.py in
    # this directory installs lightweight stubs for those modules before any
    # test module is collected. Putting test/ on sys.path mirrors the
    # documented `PYTHONPATH=test` invocation.
    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)

    try:
        import pytest
    except ImportError:  # pragma: no cover - environment guard
        print(
            "pytest is required to run the run_suite unit tests; "
            "install it with `pip install pytest`.",
            file=sys.stderr,
        )
        return 2

    args = [
        UNIT_DIR,
        "-q",
        "-p",
        "no:cacheprovider",
        "-c",
        os.path.join(UNIT_DIR, "pytest.ini"),
        "--rootdir",
        UNIT_DIR,
    ]
    args.extend(argv)
    return int(pytest.main(args))


if __name__ == "__main__":
    sys.exit(main())
