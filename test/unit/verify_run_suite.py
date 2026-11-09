"""CI entrypoint for the run_suite verification dimensions.

Runs the collect-diagnostics, exit-codes, tests and usability dimensions with
nothing but pytest installed (test/unit/conftest.py stubs sglang.test.ci.*).

Exit codes are intentionally distinct so CI can tell an environment problem
apart from a real dimension failure:

    0  all requested dimensions passed
    1  a dimension failed (real test failure)
    2  the environment itself is broken (pytest missing, etc.)

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

# Each dimension maps to the pytest node ids that cover it. The commands in
# test/README.md run the same files, so local runs and CI agree.
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


def _pytest_args(paths):
    return [
        *paths,
        "-q",
        "-p",
        "no:cacheprovider",
        "-c",
        os.path.join(UNIT_DIR, "pytest.ini"),
        "--rootdir",
        UNIT_DIR,
    ]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)

    try:
        import pytest
    except ImportError:  # pragma: no cover - environment guard
        print(
            "pytest is required to verify run_suite; install the minimal "
            "dependency set with `bash test/unit/setup_env.sh` "
            "(or `python -m pip install -r test/unit/requirements.txt`).",
            file=sys.stderr,
        )
        return EXIT_ENV_FAILURE

    print(f"[verify] pytest {pytest.__version__}", flush=True)

    requested = argv or list(DIMENSIONS)
    unknown = [d for d in requested if d not in DIMENSIONS]
    if unknown:
        print(
            f"[verify] unknown dimension(s): {', '.join(unknown)}; "
            f"known: {', '.join(DIMENSIONS)}",
            file=sys.stderr,
        )
        return EXIT_ENV_FAILURE

    failed = []
    for dimension in requested:
        paths = DIMENSIONS[dimension]
        args = _pytest_args(paths)
        print(
            f"[verify] dimension={dimension} command="
            f"python -m pytest {' '.join(args)}",
            flush=True,
        )
        code = int(pytest.main(args))
        print(f"[verify] dimension={dimension} exit={code}", flush=True)
        if code != 0:
            failed.append(dimension)

    if failed:
        print(f"[verify] failed dimensions: {', '.join(failed)}", file=sys.stderr)
        return EXIT_DIMENSION_FAILURE

    print("[verify] all dimensions passed", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
