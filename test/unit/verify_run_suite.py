"""Pytest-only verification entrypoint for run_suite.py.

This script is the CI entrypoint for the run_suite exit-code and
collection-diagnostic checks. It must be runnable with only pytest
installed: it never imports sglang, and pytest is imported lazily so
``--help`` and the environment banner work even when pytest is missing.

Exit codes:
    0  all dimensions passed
    1  a real dimension (test) failure
    2  infrastructure failure (pytest missing, test file missing, ...)

Usage (from the repository root):

    python -m pip install -r test/unit/requirements.txt
    python test/unit/verify_run_suite.py
"""

import argparse
import os
import sys


UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)

EXIT_OK = 0
EXIT_DIMENSION_FAILURE = 1
EXIT_INFRA_FAILURE = 2

# Each dimension maps to the test files that cover it. Keeping the mapping
# explicit means a CI job cannot silently run zero tests for a dimension.
DIMENSIONS = {
    "collect-diagnostics": [
        os.path.join(UNIT_DIR, "test_run_suite_collect_diagnostics.py"),
    ],
    "exit-codes": [
        os.path.join(UNIT_DIR, "test_run_suite_exit_codes.py"),
    ],
    "tests": [
        os.path.join(UNIT_DIR, "test_run_suite_exit_codes.py"),
        os.path.join(UNIT_DIR, "test_run_suite_collect_diagnostics.py"),
    ],
    "usability": [
        os.path.join(UNIT_DIR, "test_run_suite_unit_config.py"),
    ],
}


def _print_environment() -> None:
    print(f"python: {sys.version.split()[0]}", flush=True)
    try:
        import pytest  # noqa: F401

        print(f"pytest: {pytest.__version__}", flush=True)
    except ImportError:
        print("pytest: NOT INSTALLED", flush=True)


def _check_targets() -> list:
    missing = []
    for name, targets in DIMENSIONS.items():
        if not targets:
            missing.append(f"dimension {name} has no targets")
        for target in targets:
            if not os.path.exists(target):
                missing.append(f"{name}: missing {target}")
    return missing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the pytest-only run_suite verification dimensions."
    )
    parser.add_argument(
        "--dimension",
        action="append",
        choices=sorted(DIMENSIONS),
        help="Run only the named dimension(s); default: all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List dimensions and their test files, then exit.",
    )
    args = parser.parse_args(argv)

    _print_environment()

    if args.list:
        for name, targets in DIMENSIONS.items():
            print(f"{name}:")
            for target in targets:
                print(f"  {target}")
        return EXIT_OK

    missing = _check_targets()
    if missing:
        print("INFRA FAILURE: verification targets are missing:", flush=True)
        for item in missing:
            print(f"  - {item}", flush=True)
        return EXIT_INFRA_FAILURE

    try:
        import pytest
    except ImportError:
        print(
            "INFRA FAILURE: pytest is not installed; run "
            "`python -m pip install -r test/unit/requirements.txt`.",
            flush=True,
        )
        return EXIT_INFRA_FAILURE

    selected = args.dimension or list(DIMENSIONS)
    targets = []
    for name in selected:
        for target in DIMENSIONS[name]:
            if target not in targets:
                targets.append(target)

    # run_suite.py lives in test/ and imports sglang.test.ci.*; conftest.py in
    # this directory installs lightweight stubs before collection.
    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)

    pytest_args = [
        *targets,
        "-q",
        "-rs",
        "-p",
        "no:cacheprovider",
        "-c",
        os.path.join(UNIT_DIR, "pytest.ini"),
        "--rootdir",
        UNIT_DIR,
    ]
    rc = int(pytest.main(pytest_args))
    if rc == 0:
        return EXIT_OK
    # pytest exit code 5 means "no tests collected": that is an infra problem
    # for this entrypoint, not a real dimension failure.
    if rc == 5:
        print("INFRA FAILURE: no tests were collected.", flush=True)
        return EXIT_INFRA_FAILURE
    return EXIT_DIMENSION_FAILURE


if __name__ == "__main__":
    sys.exit(main())
