"""Pytest-only verification entrypoint for run_suite.py diagnostics.

This script is the CI entrypoint for the run_suite verification dimensions
(collect-diagnostics, exit-codes, tests, usability). It deliberately avoids
importing sglang: the unit tests under test/unit stub out sglang.test.ci.* in
conftest.py, so only pytest is required.

Exit codes:

* ``0`` -- every dimension passed.
* ``1`` -- at least one dimension failed (a real test failure).
* ``2`` -- the environment itself is broken (pytest missing, config missing),
  so the failure must not be reported as a test failure.

Usage (from the repository root):

    python -m pip install -r test/unit/requirements.txt
    python test/unit/verify_run_suite.py
"""

import os
import sys


UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)

EXIT_OK = 0
EXIT_TEST_FAILURE = 1
EXIT_INFRA_FAILURE = 2

# Each dimension maps to the pytest target(s) that exercise it. The
# collect-diagnostics dimension runs a real `--collect-only` over test/ so a
# collection error surfaces instead of being swallowed; the other dimensions
# run the focused unit tests.
DIMENSIONS = [
    (
        "collect-diagnostics",
        [
            TEST_DIR,
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
    ),
    (
        "exit-codes",
        [
            os.path.join(UNIT_DIR, "test_run_suite_exit_codes.py"),
            "-q",
            "-p",
            "no:cacheprovider",
        ],
    ),
    (
        "tests",
        [
            os.path.join(UNIT_DIR, "test_run_suite_exit_codes.py"),
            os.path.join(UNIT_DIR, "test_run_suite_collect_diagnostics.py"),
            "-q",
            "-p",
            "no:cacheprovider",
        ],
    ),
    (
        "usability",
        [
            os.path.join(UNIT_DIR, "test_run_suite_unit_config.py"),
            "-q",
            "-p",
            "no:cacheprovider",
        ],
    ),
]


def _print_pytest_version() -> None:
    try:
        import pytest
    except ImportError:  # pragma: no cover - guarded by main()
        return
    print(f"pytest version: {pytest.__version__}", flush=True)


def _run_dimension(name, pytest_args, pytest_main) -> int:
    cmd = "python -m pytest " + " ".join(pytest_args)
    print(f"[{name}] running: {cmd}", flush=True)
    code = int(pytest_main(pytest_args))
    print(f"[{name}] exit code: {code}", flush=True)
    return code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # run_suite.py lives in test/ and imports sglang.test.ci.*; conftest.py in
    # test/unit installs lightweight stubs before any test module is collected.
    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)

    try:
        import pytest
    except ImportError:  # pragma: no cover - environment guard
        print(
            "pytest is required to verify run_suite.py; install the minimal "
            "dependency set with "
            "`python -m pip install -r test/unit/requirements.txt` "
            "(or `pip install pytest`).",
            file=sys.stderr,
            flush=True,
        )
        return EXIT_INFRA_FAILURE

    _print_pytest_version()

    # Only run the dimensions requested on the command line (if any), so CI
    # can map one job per dimension while sharing this entrypoint.
    selected = [d for d in DIMENSIONS if not argv or d[0] in argv]
    if not selected:
        print(
            f"unknown dimension(s): {argv}; known: "
            f"{[d[0] for d in DIMENSIONS]}",
            file=sys.stderr,
            flush=True,
        )
        return EXIT_INFRA_FAILURE

    failed = []
    for name, pytest_args in selected:
        code = _run_dimension(name, pytest_args, pytest.main)
        if code != 0:
            failed.append((name, code))

    if failed:
        for name, code in failed:
            print(f"dimension failed: {name} (exit {code})", flush=True)
        return EXIT_TEST_FAILURE

    print("all dimensions passed", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
