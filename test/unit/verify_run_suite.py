"""Lightweight CI entrypoint for the run_suite verification dimensions.

This script is intentionally dependency-free: it only imports the standard
library at module scope and imports pytest lazily. CI verification jobs for
the run_suite exit-code / collection-diagnostic work must be runnable with
nothing but pytest installed -- installing the full sglang dependency set
(`pip install -e python`) previously failed during environment preparation
and prevented any dimension from running.

Usage (from the repository root):

    python test/unit/verify_run_suite.py
    python test/unit/verify_run_suite.py --dimension collect-diagnostics
    python test/unit/verify_run_suite.py --dimension exit-codes
    python test/unit/verify_run_suite.py --dimension tests
    python test/unit/verify_run_suite.py --dimension usability

Exit codes:

    0  every selected dimension passed
    1  a real dimension failure (a test or check failed)
    2  infrastructure failure (pytest missing, wrong interpreter, ...)

The 1-vs-2 split keeps an environment problem from being reported as a code
failure, which is exactly the confusion this work is meant to remove.
"""

import argparse
import os
import sys

EXIT_OK = 0
EXIT_DIMENSION_FAILURE = 1
EXIT_INFRA_FAILURE = 2

UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)
REPO_ROOT = os.path.dirname(TEST_DIR)

# Each dimension maps to the pytest node ids that exercise it. Keeping the
# mapping here (rather than in CI YAML) means the four dimensions documented
# in test/README.md are the same four this entrypoint runs.
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
        os.path.join(UNIT_DIR, "test_run_suite_standalone.py"),
        os.path.join(UNIT_DIR, "test_run_suite_no_sglang_smoke.py"),
    ],
    "usability": [
        os.path.join(UNIT_DIR, "test_run_suite_unit_config.py"),
        os.path.join(UNIT_DIR, "test_run_suite_exit_codes.py"),
    ],
}


def _print_environment() -> None:
    """Print interpreter / pytest versions so environment issues are visible."""
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    try:
        import pytest  # noqa: F401

        print(f"pytest: {pytest.__version__}")
    except ImportError:
        print("pytest: NOT INSTALLED")


def _run_dimension(name: str, extra_args) -> int:
    """Run one dimension via pytest; return 0/1/2 per the exit-code contract."""
    try:
        import pytest
    except ImportError:
        print(
            "pytest is required to run the run_suite verification dimensions; "
            "install the minimal dependency set with "
            "`python -m pip install -r test/unit/requirements.txt` "
            "(or `pip install pytest`).",
            file=sys.stderr,
        )
        return EXIT_INFRA_FAILURE

    targets = DIMENSIONS[name]
    missing = [t for t in targets if not os.path.exists(t)]
    if missing:
        print(
            f"dimension {name!r} references missing test file(s): {missing}",
            file=sys.stderr,
        )
        return EXIT_INFRA_FAILURE

    # run_suite.py lives in test/ and imports sglang.test.ci.*; conftest.py in
    # this directory installs lightweight stubs before collection, so putting
    # test/ on sys.path mirrors the documented `PYTHONPATH=test` invocation.
    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)

    args = [
        *targets,
        "-q",
        "-p",
        "no:cacheprovider",
        "-c",
        os.path.join(UNIT_DIR, "pytest.ini"),
        "--rootdir",
        UNIT_DIR,
    ]
    args.extend(extra_args)
    print(f"=== dimension: {name} ===", flush=True)
    rc = int(pytest.main(args))
    # pytest exit codes: 0 ok, 1 tests failed, 2 interrupted, 3 internal error,
    # 4 usage error, 5 no tests collected. Only 0 is success; 2/3/4/5 are
    # environment/usage problems, everything else is a real failure.
    if rc == 0:
        return EXIT_OK
    if rc in (2, 3, 4, 5):
        return EXIT_INFRA_FAILURE
    return EXIT_DIMENSION_FAILURE


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run the run_suite verification dimensions (pytest-only)."
    )
    parser.add_argument(
        "--dimension",
        choices=sorted(DIMENSIONS),
        action="append",
        default=None,
        help="Dimension to run; repeatable. Defaults to all four.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to pytest.",
    )
    args = parser.parse_args(argv)

    _print_environment()

    selected = args.dimension or list(DIMENSIONS)
    worst = EXIT_OK
    for name in selected:
        rc = _run_dimension(name, args.pytest_args)
        print(f"=== dimension {name}: exit {rc} ===", flush=True)
        # Infra failure dominates: it means the dimension never really ran.
        if rc == EXIT_INFRA_FAILURE:
            return EXIT_INFRA_FAILURE
        if rc != EXIT_OK:
            worst = EXIT_DIMENSION_FAILURE
    return worst


if __name__ == "__main__":
    sys.exit(main())
