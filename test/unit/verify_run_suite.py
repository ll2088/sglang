"""CI entrypoint for the run_suite verification dimensions.

Runs the collect-diagnostics, exit-codes, tests and usability dimensions with
nothing but pytest installed. Exit codes are distinct so an environment
problem is never confused with a real test failure:

    0  all dimensions passed
    1  at least one dimension failed
    2  the environment itself is broken (pytest missing / not importable)

Usage (from the repository root):

    python test/unit/verify_run_suite.py
"""

import os
import sys


UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)

EXIT_OK = 0
EXIT_DIMENSION_FAILURE = 1
EXIT_ENV_FAILURE = 2

# Each dimension maps to the pytest targets that exercise it. The exit-code
# and collect-diagnostic tests are the same files the acceptance criteria name,
# so a single run covers both dimensions without duplicating work.
DIMENSIONS = [
    (
        "collect-diagnostics",
        [
            os.path.join(UNIT_DIR, "test_run_suite_collect_diagnostics.py"),
        ],
    ),
    (
        "exit-codes",
        [
            os.path.join(UNIT_DIR, "test_run_suite_exit_codes.py"),
        ],
    ),
    (
        "tests",
        [
            os.path.join(UNIT_DIR, "test_run_suite_exit_codes.py"),
            os.path.join(UNIT_DIR, "test_run_suite_collect_diagnostics.py"),
        ],
    ),
    (
        "usability",
        [
            os.path.join(UNIT_DIR, "test_run_suite_unit_config.py"),
        ],
    ),
]


def _run_dimension(pytest, name, targets):
    print(f"[verify] dimension: {name}", flush=True)
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
    code = int(pytest.main(args))
    print(f"[verify] dimension {name}: exit {code}", flush=True)
    return code


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)

    try:
        import pytest
    except ImportError:
        print(
            "pytest is required to verify run_suite; install the minimal "
            "dependency set with `bash test/unit/setup_env.sh` "
            "(or `python -m pip install -r test/unit/requirements.txt`).",
            file=sys.stderr,
        )
        return EXIT_ENV_FAILURE

    print(f"[verify] pytest {pytest.__version__}", flush=True)

    failed = []
    for name, targets in DIMENSIONS:
        missing = [t for t in targets if not os.path.exists(t)]
        if missing:
            print(f"[verify] dimension {name}: missing {missing}", file=sys.stderr)
            failed.append(name)
            continue
        if _run_dimension(pytest, name, targets) != 0:
            failed.append(name)

    if failed:
        print(f"[verify] failed dimensions: {', '.join(failed)}", file=sys.stderr)
        return EXIT_DIMENSION_FAILURE

    print("[verify] all dimensions passed", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
