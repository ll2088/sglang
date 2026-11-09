"""CI entrypoint for the run_suite.py verification dimensions.

This script is intentionally dependency-free beyond pytest: the run_suite
unit tests stub out ``sglang.test.ci.*`` in ``conftest.py``, so the full
sglang runtime (torch, kernels, ...) is not needed. CI verification jobs must
install only ``test/unit/requirements.txt`` instead of ``pip install -e
python``, which resolves the full dependency set and can fail during
environment preparation.

Exit codes:

* ``0`` -- every dimension passed.
* ``1`` -- at least one dimension failed (a real test failure).
* ``2`` -- the environment itself is broken (pytest missing, etc.), so an
  infra problem is never confused with a test failure.

Usage (from the repository root):

    python -m pip install -r test/unit/requirements.txt
    python test/unit/verify_run_suite.py
"""

import os
import platform
import subprocess
import sys


UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)
REPO_ROOT = os.path.dirname(TEST_DIR)

EXIT_OK = 0
EXIT_DIMENSION_FAILURE = 1
EXIT_INFRA_FAILURE = 2

# Each dimension maps to the pytest targets that exercise it. The tests are
# the same files the acceptance criteria name, so a dimension failure points
# at the exact behaviour that regressed.
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
            os.path.join(UNIT_DIR, "test_run_suite_exit_codes.py"),
        ],
    ),
]


def _print_environment() -> None:
    """Print the toolchain versions so an env failure is diagnosable."""
    print(f"python: {sys.version.split()[0]} ({platform.platform()})", flush=True)
    print(f"executable: {sys.executable}", flush=True)
    try:
        pip_version = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"pip: {pip_version.stdout.strip() or pip_version.stderr.strip()}", flush=True)
    except OSError as exc:  # pragma: no cover - environment guard
        print(f"pip: unavailable ({exc})", flush=True)
    try:
        import pytest  # noqa: F401

        print(f"pytest: {pytest.__version__}", flush=True)
    except ImportError:  # pragma: no cover - environment guard
        print("pytest: NOT INSTALLED", flush=True)


def _run_dimension(name: str, targets: list) -> int:
    """Run one dimension and return its pytest exit code."""
    print(f"\n=== dimension: {name} ===", flush=True)
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


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # run_suite.py lives in test/ and imports sglang.test.ci.*; conftest.py in
    # this directory installs lightweight stubs for those modules before any
    # test module is collected. Putting test/ on sys.path mirrors the
    # documented `PYTHONPATH=test` invocation.
    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)

    _print_environment()

    try:
        import pytest
    except ImportError:  # pragma: no cover - environment guard
        print(
            "pytest is required to verify run_suite; install the minimal "
            "dependency set with "
            "`python -m pip install -r test/unit/requirements.txt` "
            "(or `pip install pytest`).",
            file=sys.stderr,
            flush=True,
        )
        return EXIT_INFRA_FAILURE

    # Expose pytest to _run_dimension without a module-level import so the
    # missing-pytest guard above stays the single infra check.
    globals()["pytest"] = pytest

    failed = []
    for name, targets in DIMENSIONS:
        code = _run_dimension(name, targets)
        if code != 0:
            failed.append((name, code))

    print("\n=== summary ===", flush=True)
    if failed:
        for name, code in failed:
            print(f"FAILED {name} (pytest exit {code})", flush=True)
        return EXIT_DIMENSION_FAILURE
    print("all dimensions passed", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
