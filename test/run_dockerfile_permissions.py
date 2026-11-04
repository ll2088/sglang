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


if __name__ == "__main__":
    sys.exit(_run_standalone())
