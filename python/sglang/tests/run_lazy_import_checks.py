"""Run the lazy-import error checks without requiring pytest.

CI environments may not have pytest installed. This entry point runs the
standard-library verification script and, when pytest is available, also
runs the pytest-based test module. It exits non-zero on any failure so it
can be used directly as a CI command.

Usage::

    python python/sglang/tests/run_lazy_import_checks.py
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _run_stdlib_verification() -> int:
    script = _HERE / "verify_lazy_import_errors.py"
    if not script.is_file():
        print(f"missing verification script: {script}", file=sys.stderr)
        return 1
    result = subprocess.run([sys.executable, str(script)], check=False)
    return result.returncode


def _run_pytest_if_available() -> int:
    if importlib.util.find_spec("pytest") is None:
        print("pytest not installed; skipping pytest-based lazy import tests")
        return 0
    test_file = _HERE / "test_lazy_import.py"
    if not test_file.is_file():
        print(f"missing pytest test file: {test_file}", file=sys.stderr)
        return 1
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", str(test_file)],
        check=False,
    )
    return result.returncode


def main() -> int:
    rc = _run_stdlib_verification()
    if rc != 0:
        return rc
    return _run_pytest_if_available()


if __name__ == "__main__":
    sys.exit(main())
