"""Pytest-only verification entrypoint for run_suite.py.

CI previously prepared the environment with ``pip install -e python``, which
resolves the full sglang dependency set and can fail before pytest ever
starts. This script is the minimal-dependency entrypoint: it only needs
pytest (see ``test/unit/requirements.txt``) and runs the four verification
dimensions used to validate run_suite.py's collection diagnostics and exit
codes.

Exit codes:
    0  all dimensions passed
    1  at least one dimension failed (real test failure)
    2  environment/infra failure (pytest missing, import error, ...)

Usage (from the repository root):

    python -m pip install -r test/unit/requirements.txt
    python test/unit/verify_run_suite.py
"""

import os
import subprocess
import sys


UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)
REPO_ROOT = os.path.dirname(TEST_DIR)

EXIT_OK = 0
EXIT_DIMENSION_FAILURE = 1
EXIT_INFRA_FAILURE = 2

# Each dimension maps to the pytest invocation that exercises it. The
# collect-diagnostics dimension uses --collect-only over test/ so a broken
# collection surfaces as a non-zero exit instead of being swallowed.
DIMENSIONS = [
    (
        "collect-diagnostics",
        [
            sys.executable,
            "-m",
            "pytest",
            "test",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
    ),
    (
        "exit-codes",
        [
            sys.executable,
            "-m",
            "pytest",
            "test/unit/test_run_suite_exit_codes.py",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
    ),
    (
        "tests",
        [
            sys.executable,
            "-m",
            "pytest",
            "test/unit/test_run_suite_exit_codes.py",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
    ),
    (
        "usability",
        [
            sys.executable,
            "-m",
            "pytest",
            "test/unit/test_run_suite_exit_codes.py",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
    ),
]


def _env():
    env = dict(os.environ)
    # run_suite.py lives in test/ and imports sglang.test.ci.*; conftest.py
    # installs lightweight stubs, so putting test/ on PYTHONPATH mirrors the
    # documented invocation and keeps the full runtime out of the picture.
    env["PYTHONPATH"] = os.pathsep.join(
        [TEST_DIR, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    return env


def _check_pytest():
    try:
        import pytest  # noqa: F401
    except ImportError:
        print(
            "pytest is required; install the minimal dependency set with "
            "`python -m pip install -r test/unit/requirements.txt`.",
            file=sys.stderr,
        )
        return False
    return True


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if not _check_pytest():
        return EXIT_INFRA_FAILURE

    import pytest

    print(f"pytest version: {pytest.__version__}", flush=True)
    print(f"python: {sys.version.split()[0]}", flush=True)

    env = _env()
    failed = []
    for name, cmd in DIMENSIONS:
        print(f"\n=== dimension: {name} ===", flush=True)
        print("$ " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        if proc.returncode != 0:
            failed.append(name)

    if failed:
        print(
            "\nFAILED dimensions: " + ", ".join(failed),
            file=sys.stderr,
            flush=True,
        )
        return EXIT_DIMENSION_FAILURE

    print("\nAll dimensions passed.", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
