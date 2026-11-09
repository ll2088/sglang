"""CI entrypoint for the run_suite verification dimensions.

Runs the collect-diagnostics, exit-codes, tests and usability dimensions with
pytest only (no full sglang runtime). Exits 2 when the environment itself is
broken (infra failure) versus 1 for a real dimension failure, so an
environment problem is never confused with a test failure.

Usage (from the repository root):

    bash test/unit/setup_env.sh
    python test/unit/verify_run_suite.py
"""

import argparse
import os
import subprocess
import sys

UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)
REPO_ROOT = os.path.dirname(TEST_DIR)

EXIT_OK = 0
EXIT_DIMENSION_FAILURE = 1
EXIT_INFRA_FAILURE = 2

# Each dimension maps to the pytest targets that exercise it. Every target
# must exist so a CI job cannot silently run zero tests for a dimension.
DIMENSIONS = {
    "collect-diagnostics": [
        os.path.join(UNIT_DIR, "test_run_suite_collect_diagnostics.py"),
    ],
    "exit-codes": [
        os.path.join(UNIT_DIR, "test_run_suite_exit_codes.py"),
    ],
    "tests": [
        os.path.join(UNIT_DIR, "test_run_suite_exit_codes.py"),
        os.path.join(UNIT_DIR, "test_run_suite_standalone.py"),
        os.path.join(UNIT_DIR, "test_run_suite_no_sglang_smoke.py"),
    ],
    "usability": [
        os.path.join(UNIT_DIR, "test_run_suite_unit_config.py"),
    ],
}


def _pytest_available() -> bool:
    try:
        import pytest  # noqa: F401
    except ImportError:
        return False
    return True


def _run_dimension(name: str, targets: list) -> int:
    print(f"[verify] dimension={name} targets={len(targets)}", flush=True)
    for target in targets:
        print(f"[verify]   {target}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [TEST_DIR, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-q",
        "-p",
        "no:cacheprovider",
        "-c",
        os.path.join(UNIT_DIR, "pytest.ini"),
        "--rootdir",
        UNIT_DIR,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return proc.returncode


def main(argv=None) -> int:
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

    print(f"[verify] repo_root={REPO_ROOT}", flush=True)
    print(f"[verify] unit_dir={UNIT_DIR}", flush=True)

    if not _pytest_available():
        print(
            "[verify] pytest NOT INSTALLED; run "
            "`bash test/unit/setup_env.sh` first.",
            file=sys.stderr,
            flush=True,
        )
        return EXIT_INFRA_FAILURE

    import pytest

    print(f"[verify] pytest {pytest.__version__}", flush=True)

    selected = args.dimension or list(DIMENSIONS)
    failures = []
    for name in selected:
        targets = DIMENSIONS[name]
        missing = [t for t in targets if not os.path.exists(t)]
        if missing:
            print(f"[verify] {name}: missing targets {missing}", file=sys.stderr)
            return EXIT_INFRA_FAILURE
        code = _run_dimension(name, targets)
        if code != 0:
            failures.append((name, code))

    if failures:
        for name, code in failures:
            print(f"[verify] FAILED {name} (exit {code})", file=sys.stderr)
        return EXIT_DIMENSION_FAILURE

    print("[verify] all dimensions passed", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
