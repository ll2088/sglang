"""Tests for the pytest-only verification entrypoint.

These tests exercise verify_run_suite.py's dimension dispatch and its
infra-vs-test exit-code contract without requiring the full sglang runtime.
"""

import importlib.util
import os
import subprocess
import sys

UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)
REPO_ROOT = os.path.dirname(TEST_DIR)
ENTRYPOINT = os.path.join(UNIT_DIR, "verify_run_suite.py")


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "verify_run_suite_entrypoint_under_test", ENTRYPOINT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entrypoint_help_runs_without_pytest_import():
    """`--help` must work even if pytest is unavailable (lazy import)."""
    proc = subprocess.run(
        [sys.executable, ENTRYPOINT, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--dimension" in proc.stdout
    for dimension in (
        "collect-diagnostics",
        "exit-codes",
        "tests",
        "usability",
    ):
        assert dimension in proc.stdout


def test_entrypoint_reports_missing_pytest_as_infra_failure():
    """A missing pytest must exit 2 (infra), never 1 (test failure)."""
    script = (
        "import sys, types\n"
        "sys.modules['pytest'] = None\n"
        "sys.path.insert(0, %r)\n"
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('v', %r)\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "rc = m._run_dimension('exit-codes', [])\n"
        "assert rc == m.EXIT_INFRA_FAILURE, rc\n"
        "print('INFRA_OK')\n"
    ) % (UNIT_DIR, ENTRYPOINT)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "INFRA_OK" in proc.stdout


def test_entrypoint_missing_target_is_infra_failure(monkeypatch):
    """A dimension pointing at a missing file is an infra failure, not a test."""
    module = _load_entrypoint()
    monkeypatch.setitem(
        module.DIMENSIONS, "exit-codes", [os.path.join(UNIT_DIR, "does_not_exist.py")]
    )
    rc = module._run_dimension("exit-codes", [])
    assert rc == module.EXIT_INFRA_FAILURE


def test_entrypoint_exit_codes_are_distinct():
    module = _load_entrypoint()
    codes = {
        module.EXIT_OK,
        module.EXIT_DIMENSION_FAILURE,
        module.EXIT_INFRA_FAILURE,
    }
    assert len(codes) == 3
    assert module.EXIT_OK == 0
    assert module.EXIT_DIMENSION_FAILURE == 1
    assert module.EXIT_INFRA_FAILURE == 2
