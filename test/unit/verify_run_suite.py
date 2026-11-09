"""Minimal, dependency-light verification entrypoint for run_suite.py.

CI previously prepared the environment with ``pip install -e python``, which
resolves the full sglang dependency set and can fail before pytest ever runs,
turning an infrastructure problem into a misleading dimension failure. This
script is the single command CI can call instead:

    python test/unit/verify_run_suite.py

It only needs pytest (see ``test/unit/requirements.txt``) and runs the four
verification dimensions in order, printing the pytest version and the
installed dependency list up front so an environment failure is easy to tell
apart from a logic failure.

Exit codes:

* ``0``  -- all dimensions passed
* ``1``  -- at least one dimension failed (logic failure)
* ``2``  -- environment preparation failed (infra failure)
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

# (dimension name, pytest args). Each dimension is a plain pytest invocation so
# the same commands documented in test/README.md keep working standalone.
DIMENSIONS = [
    (
        "collect-diagnostics",
        [
            os.path.join(TEST_DIR, "unit", "test_run_suite_collect_diagnostics.py"),
            os.path.join(TEST_DIR, "unit", "test_run_suite_exit_codes.py"),
        ],
    ),
    (
        "exit-codes",
        [os.path.join(TEST_DIR, "unit", "test_run_suite_exit_codes.py")],
    ),
    (
        "tests",
        [
            os.path.join(TEST_DIR, "unit", "test_run_suite_exit_codes.py"),
            os.path.join(TEST_DIR, "unit", "test_run_suite_collect_diagnostics.py"),
            os.path.join(TEST_DIR, "unit", "test_run_suite_unit_config.py"),
        ],
    ),
    (
        "usability",
        [
            os.path.join(TEST_DIR, "unit", "test_run_suite_unit_config.py"),
            os.path.join(TEST_DIR, "unit", "test_run_suite_exit_codes.py"),
        ],
    ),
]


def _print_environment() -> None:
    """Print pytest version and installed packages for triage."""
    print("=" * 70, flush=True)
    print("ENVIRONMENT", flush=True)
    print(f"  python: {sys.version.split()[0]}", flush=True)
    try:
        import pytest  # noqa: F401

        print(f"  pytest: {pytest.__version__}", flush=True)
    except ImportError:
        print("  pytest: NOT INSTALLED", flush=True)
    try:
        from importlib import metadata

        dists = sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in metadata.distributions()
            if d.metadata.get("Name")
        )
        print(f"  installed packages ({len(dists)}):", flush=True)
        for dist in dists:
            print(f"    {dist}", flush=True)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never crash
        print(f"  installed packages: unavailable ({exc})", flush=True)
    print("=" * 70, flush=True)


def _run_dimension(name: str, targets: list) -> int:
    print(f"\n--- dimension: {name} ---", flush=True)
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
    print("  $ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return proc.returncode


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    _print_environment()

    try:
        import pytest  # noqa: F401
    except ImportError:
        print(
            "pytest is required; install the minimal dependency set with "
            "`python -m pip install -r test/unit/requirements.txt`.",
            file=sys.stderr,
            flush=True,
        )
        return EXIT_INFRA_FAILURE

    # The tests dimension must be deterministic: run it twice and compare.
    failures = []
    for name, targets in DIMENSIONS:
        runs = 2 if name == "tests" else 1
        codes = [_run_dimension(name, targets) for _ in range(runs)]
        if any(code != 0 for code in codes):
            failures.append(name)
        elif len(set(codes)) != 1:
            failures.append(name)

    if failures:
        print(f"\nFAILED dimensions: {', '.join(failures)}", flush=True)
        return EXIT_DIMENSION_FAILURE

    print("\nALL DIMENSIONS PASSED", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
