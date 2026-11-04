#!/usr/bin/env python3
"""Stdlib-only entrypoint for the Dockerfile permission checks.

This module is a thin wrapper around ``test/dockerfile_permissions/checks.py``
so the checks can be run either directly with a bare ``python3`` interpreter
or collected by pytest without importing sglang or installing its full
dependency tree.

Usage:
    python3 test/test_dockerfile_permissions.py
    python3 -m pytest -q test/test_dockerfile_permissions.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHECKS_DIR = os.path.join(_HERE, "dockerfile_permissions")
if _CHECKS_DIR not in sys.path:
    sys.path.insert(0, _CHECKS_DIR)

from checks import ALL_CHECKS  # noqa: E402  (path set up above)


def test_all_checks_pass():
    """Run every check so a single pytest invocation covers the full suite."""
    for check in ALL_CHECKS:
        check()


def main():
    """Run every check under a bare python3 interpreter."""
    failures = []
    for check in ALL_CHECKS:
        try:
            check()
        except AssertionError as exc:
            failures.append((check.__name__, exc))
    if failures:
        for name, exc in failures:
            print(f"FAIL: {name}: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {len(ALL_CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
