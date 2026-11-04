#!/usr/bin/env python3
"""Standalone runner for the stdlib-only Dockerfile permission checks.

Usage:
    python3 test/dockerfile_permissions/run.py

This entry point does not import sglang or pytest, so it can run in a bare
python3 environment.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from checks import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
