#!/usr/bin/env python3
"""Standalone entry point for the stdlib-only Dockerfile permission checks.

This runner imports nothing outside the Python standard library and never
imports sglang, so CI can execute the permission/documentation checks without
installing the full sglang dependency tree:

    python3 test/dockerfile_permissions/run.py

It delegates to the checks in ``test_dockerfile_permissions.py`` and exits
non-zero if any check fails.
"""

import sys
from pathlib import Path

# Make the sibling test module importable regardless of the caller's CWD.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import test_dockerfile_permissions as checks  # noqa: E402


def main() -> int:
    missing = checks.missing_paths()
    if missing:
        print("missing paths:")
        for item in missing:
            print(f"  {item}")
        return 1

    failures = 0
    for check in checks.CHECKS:
        try:
            check()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {check.__name__}: {exc}")
        else:
            print(f"PASS {check.__name__}")

    print(f"readme_keyword_hits_total={checks.readme_keyword_hit_count()}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
