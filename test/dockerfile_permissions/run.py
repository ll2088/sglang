#!/usr/bin/env python3
"""Standalone runner for the stdlib-only Dockerfile permission checks.

Usage:
    python3 test/dockerfile_permissions/run.py

Exits non-zero if any check fails. Requires only the Python standard library.
"""

import sys

from checks import ALL_CHECKS


def main():
    failures = 0
    for check in ALL_CHECKS:
        try:
            check()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {check.__name__}: {exc}", file=sys.stderr)
        else:
            print(f"PASS {check.__name__}")
    if failures:
        print(f"{failures} check(s) failed", file=sys.stderr)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
