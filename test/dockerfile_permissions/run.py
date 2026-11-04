#!/usr/bin/env python3
"""Standalone runner for the stdlib-only Dockerfile permission checks.

Usage:
    python3 test/dockerfile_permissions/run.py

This entry point requires only the Python standard library and never imports
sglang or pytest, so CI can run it without installing the full dependency tree.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from checks import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
