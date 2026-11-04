#!/usr/bin/env python3
"""Standalone runner for the stdlib-only Dockerfile permission checks.

Usage:
    python3 test/dockerfile_permissions/run.py

This entry point imports nothing outside the standard library so CI can run it
with a bare ``python3`` interpreter, without installing sglang or pytest.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from checks import run_all  # noqa: E402


if __name__ == "__main__":
    run_all()
    print("dockerfile permission checks passed")
