"""CI entrypoint for the run_suite verification dimensions.

Runs the collect-diagnostics, exit-codes, tests and usability dimensions with
nothing but pytest installed. Exits 2 when the environment itself is broken
(infra failure) and 1 when a real dimension fails, so an environment problem
is never confused with a test failure.

Usage (from the repository root):

    bash test/unit/setup_env.sh
    PYTHONPATH=test python test/unit/verify_run_suite.py
"""

import os
import sys

EXIT_OK = 0
EXIT_DIMENSION_FAILURE = 1
EXIT_INFRA_FAILURE = 2

UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)
REPO_ROOT = os.path.dirname(TEST_DIR)


def _unit(*names):
    return [os.path.join(UNIT_DIR, name) for name in names]


DIMENSIONS = {
    "collect-diagnostics": _unit(
        "test_run_suite_collect_diagnostics.py",
        "test_run_suite_no_sglang_smoke.py",
        "test_run_suite_standalone.py",
    ),
    "exit-codes": _unit("test_run_suite_exit_codes.py"),
    "tests": _unit(
        "test_run_suite_exit_codes.py",
        "test_run_suite_collect_diagnostics.py",
    ),
    "usability": _unit(
        "test_run_suite_unit_config.py",
        "test_run_suite_exit_codes.py",
    ),
}


def _print_environment() -> bool:
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("pytest: NOT INSTALLED")
        print(
            "Install the minimal dependency set with "
            "`bash test/unit/setup_env.sh` (pytest only).",
            file=sys.stderr,
        )
        return False
    print(f"pytest: {pytest.__version__}")
    return True


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--help" in argv or "-h" in argv:
        print(__doc__)
        print("Dimensions:")
        for name, targets in DIMENSIONS.items():
            print(f"  {name}: {len(targets)} file(s)")
        return EXIT_OK

    if not _print_environment():
        return EXIT_INFRA_FAILURE

    import pytest

    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)

    failed = []
    for name, targets in DIMENSIONS.items():
        missing = [t for t in targets if not os.path.exists(t)]
        if missing:
            print(f"[{name}] missing test files: {missing}", file=sys.stderr)
            failed.append(name)
            continue
        print(f"[{name}] running {len(targets)} file(s)", flush=True)
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
        if code != 0:
            print(f"[{name}] FAILED (pytest exit {code})", file=sys.stderr)
            failed.append(name)
        else:
            print(f"[{name}] OK", flush=True)

    if failed:
        print(f"dimension failures: {', '.join(failed)}", file=sys.stderr)
        return EXIT_DIMENSION_FAILURE
    print("all dimensions passed")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
