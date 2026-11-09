"""CI entrypoint for the run_suite verification dimensions.

Runs the collect-diagnostics, exit-codes, tests and usability dimensions with
nothing but pytest installed. Exits 2 when the environment itself is broken
(infra failure) versus 1 for a real dimension failure, so CI can tell the two
apart instead of reporting a generic exit code 1.

Usage (from the repository root):

    bash test/unit/setup_env.sh
    python test/unit/verify_run_suite.py
"""

import os
import sys


UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)

EXIT_OK = 0
EXIT_DIMENSION_FAILURE = 1
EXIT_ENV_FAILURE = 2

# Each dimension maps to the pytest targets that exercise it. The commands are
# kept as pytest args (not shell strings) so the same entry works locally and
# in CI without depending on a shell.
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
    ],
}


def _check_environment():
    """Return None when the environment is usable, else a diagnostic string."""
    try:
        import pytest  # noqa: F401
    except ImportError as exc:
        return (
            "pytest is not importable: "
            f"{exc}. Run `bash test/unit/setup_env.sh` "
            "(installs test/unit/requirements.txt, pytest only)."
        )
    if not os.path.isdir(UNIT_DIR):
        return f"unit test directory missing: {UNIT_DIR}"
    return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    env_error = _check_environment()
    if env_error is not None:
        print(f"[verify] ENVIRONMENT FAILURE: {env_error}", file=sys.stderr)
        return EXIT_ENV_FAILURE

    import pytest

    print(f"[verify] pytest {pytest.__version__}")
    print(f"[verify] python {sys.version.split()[0]}")

    # run_suite.py lives in test/ and imports sglang.test.ci.*; conftest.py in
    # this directory installs lightweight stubs before any test module is
    # collected. Putting test/ on sys.path mirrors `PYTHONPATH=test`.
    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)

    selected = argv or list(DIMENSIONS)
    unknown = [name for name in selected if name not in DIMENSIONS]
    if unknown:
        print(
            f"[verify] unknown dimension(s): {', '.join(unknown)}; "
            f"known: {', '.join(DIMENSIONS)}",
            file=sys.stderr,
        )
        return EXIT_ENV_FAILURE

    failed = []
    for name in selected:
        targets = DIMENSIONS[name]
        print(f"[verify] dimension: {name}")
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
            failed.append((name, code))

    if failed:
        for name, code in failed:
            print(f"[verify] dimension failed: {name} (pytest exit {code})")
        return EXIT_DIMENSION_FAILURE

    print("[verify] all dimensions passed")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
