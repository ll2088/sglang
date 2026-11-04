#!/usr/bin/env python3
"""Standalone runner for the stdlib-only Dockerfile permission checks.

This entry point does not require pytest or sglang, so CI can run it with a
bare ``python3`` interpreter:

    python3 test/dockerfile_permissions/run.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from checks import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
