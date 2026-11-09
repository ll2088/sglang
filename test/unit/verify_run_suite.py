"""CI entrypoint for the run_suite verification dimensions.

Runs the collect-diagnostics, exit-codes, tests and usability dimensions with
pytest only. Exits 2 when the environment itself is broken (infra failure) and
1 for a real dimension failure, so CI can tell the two apart.

Usage (from the repository root):

    bash test/unit/setup_env.sh
    python test/unit/verify_run_suite.py [dimension ...]
"""

import os
import sys


EXIT_OK = 0
EXIT_DIMENSION_FAILURE = 1
EXIT_ENV_FAILURE = 2

UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)

# Each dimension maps to the pytest targets that exercise it. The unit tests
# under test/unit cover the exit-code and collection-diagnostic branches; the
# usability dimension also checks README/pytest.ini consistency.
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


def _run_pytest(targets):
    import pytest

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
    return int(pytest.main(args))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selected = argv or list(DIMENSIONS)

    unknown = [d for d in selected if d not in DIMENSIONS]
    if unknown:
        print(f"unknown dimension(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"known: {', '.join(DIMENSIONS)}", file=sys.stderr)
        return EXIT_ENV_FAILURE

    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)

    try:
        import pytest  # noqa: F401
    except ImportError:
        print(
            "pytest is required; install the minimal dependency set with "
            "`bash test/unit/setup_env.sh` (or "
            "`python -m pip install -r test/unit/requirements.txt`).",
            file=sys.stderr,
        )
        return EXIT_ENV_FAILURE

    print(f"[verify] pytest version: {pytest.__version__}")

    failed = []
    for dimension in selected:
        print(f"[verify] dimension: {dimension}")
        code = _run_pytest(DIMENSIONS[dimension])
        if code != 0:
            failed.append(dimension)

    if failed:
        print(f"[verify] failed dimension(s): {', '.join(failed)}", file=sys.stderr)
        return EXIT_DIMENSION_FAILURE
    print("[verify] all dimensions passed")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
