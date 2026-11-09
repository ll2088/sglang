"""CI entrypoint for the run_suite verification dimensions.

Runs the collect-diagnostics, exit-codes, tests and usability dimensions with
pytest only. Exits 2 when the environment itself is broken (infra failure) and
1 when a real dimension fails, so an environment problem is never confused
with a test failure.

Usage (from the repository root):

    bash test/unit/setup_env.sh
    python test/unit/verify_run_suite.py
"""

import argparse
import os
import sys

UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)
REPO_ROOT = os.path.dirname(TEST_DIR)

EXIT_OK = 0
EXIT_DIMENSION_FAILURE = 1
EXIT_INFRA_FAILURE = 2


def _unit(*names):
    return [os.path.join(UNIT_DIR, name) for name in names]


DIMENSIONS = {
    "collect-diagnostics": _unit("test_run_suite_collect_diagnostics.py"),
    "exit-codes": _unit("test_run_suite_exit_codes.py"),
    "tests": _unit("test_run_suite_exit_codes.py"),
    "usability": _unit("test_run_suite_unit_config.py"),
}


def _print_environment():
    print("[verify_run_suite] repo_root=%s" % REPO_ROOT)
    print("[verify_run_suite] unit_dir=%s" % UNIT_DIR)
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("[verify_run_suite] pytest: NOT INSTALLED")
        return False
    print("[verify_run_suite] pytest: %s" % getattr(pytest, "__version__", "unknown"))
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the run_suite verification dimensions (pytest only)."
    )
    parser.add_argument(
        "--dimension",
        action="append",
        choices=sorted(DIMENSIONS),
        help="Run only the named dimension(s); default runs all four.",
    )
    args = parser.parse_args(argv)

    if not _print_environment():
        print(
            "[verify_run_suite] install the minimal deps with "
            "`bash test/unit/setup_env.sh` (pytest only)",
            file=sys.stderr,
        )
        return EXIT_INFRA_FAILURE

    import pytest

    selected = args.dimension or list(DIMENSIONS)
    failed = []
    for name in selected:
        targets = DIMENSIONS[name]
        missing = [t for t in targets if not os.path.exists(t)]
        if missing:
            print("[verify_run_suite] %s: missing %s" % (name, missing))
            failed.append(name)
            continue
        print("[verify_run_suite] dimension=%s targets=%s" % (name, targets))
        code = pytest.main(
            [
                *targets,
                "-q",
                "-p",
                "no:cacheprovider",
                "-c",
                os.path.join(UNIT_DIR, "pytest.ini"),
                "--rootdir",
                UNIT_DIR,
            ]
        )
        if int(code) != 0:
            failed.append(name)

    if failed:
        print("[verify_run_suite] FAILED dimensions: %s" % ", ".join(failed))
        return EXIT_DIMENSION_FAILURE
    print("[verify_run_suite] OK: all dimensions passed")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
