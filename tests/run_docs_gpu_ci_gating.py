"""Entry point for GPU CI documentation checks that works with or without pytest.

Runs the dependency-free checks in ``tests/test_docs_gpu_ci_gating_stdlib.py``
and, when pytest is available, also runs the pytest-based checks in
``tests/test_docs_gpu_ci_gating.py``. Exits 0 when pytest is absent so CI
environments without pytest do not fail on a missing optional dependency.
"""

import importlib.util
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
STDLIB_TEST = ROOT / "tests" / "test_docs_gpu_ci_gating_stdlib.py"
PYTEST_TEST = ROOT / "tests" / "test_docs_gpu_ci_gating.py"


def run_stdlib():
    result = subprocess.run([sys.executable, str(STDLIB_TEST)], check=False)
    return result.returncode


def run_pytest_if_available():
    if importlib.util.find_spec("pytest") is None:
        print(
            "SKIP: pytest is not installed; stdlib documentation checks already ran",
            file=sys.stderr,
        )
        return 0
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(PYTEST_TEST)],
        check=False,
    )
    return result.returncode


def main():
    stdlib_rc = run_stdlib()
    if stdlib_rc != 0:
        return stdlib_rc
    return run_pytest_if_available()


if __name__ == "__main__":
    sys.exit(main())
