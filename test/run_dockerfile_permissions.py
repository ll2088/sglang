#!/usr/bin/env python3
"""Standalone entry point for the Dockerfile permission checks.

This wrapper exists so CI can run the checks without pytest and without
installing sglang. It imports only the stdlib-only test module and exits with
its status code.

Usage:
    python3 test/run_dockerfile_permissions.py
"""

import sys
from pathlib import Path

# Make the sibling test module importable regardless of the caller's CWD.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Importing the test module must not require pytest or sglang; the module is
# stdlib-only by design so this wrapper can run under a bare python3.
from test_dockerfile_permissions import _run_standalone  # noqa: E402


def _print_env() -> None:
    """Print the interpreter and module path so CI logs show the real entry."""
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    print(f"module: {Path(__file__).resolve()}")
    print(f"cwd: {Path.cwd()}")
    # Show whether pytest is importable so CI logs distinguish a bare-python
    # run from a pytest-collected run without installing anything extra.
    try:
        import pytest  # noqa: F401
    except Exception:
        print("pytest: not importable (stdlib-only run)")
    else:
        print("pytest: importable")


if __name__ == "__main__":
    if "--check-env" in sys.argv[1:]:
        _print_env()
        sys.exit(0)
    _print_env()
    sys.exit(_run_standalone())
