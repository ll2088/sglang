#!/usr/bin/env python3
"""Standalone runner for the stdlib-only Dockerfile permission checks.

Usage:
    python3 test/dockerfile_permissions/run.py

Exits 0 when all checks pass, 1 otherwise. Requires only the Python standard
library so CI does not need to install sglang or pytest.
"""

import sys

from checks import ALL_CHECKS


def main():
    failures = []
    for check in ALL_CHECKS:
        try:
            check()
        except AssertionError as exc:
            failures.append(f"FAIL {check.__name__}: {exc}")
        else:
            print(f"PASS {check.__name__}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
