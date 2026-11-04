#!/usr/bin/env python3
"""Standalone runner for the stdlib-only Dockerfile permission checks.

Usage::

    python3 test/dockerfile_permissions/run.py

Exits non-zero if any check fails. Requires only the Python standard library.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly (``python3 test/dockerfile_permissions/run.py``)
# without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checks import run_all  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_all())
