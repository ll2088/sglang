#!/usr/bin/env python3
"""Standalone runner for the stdlib-only Dockerfile permission checks.

This entry point exists so CI can run the permission and documentation checks
without installing sglang or any third-party package. It imports the checks
from ``test_dockerfile_permissions.py`` and exits non-zero on the first
failure, printing a short diagnostic for each failed check.

Usage:
    python3 test/dockerfile_permissions/run.py
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
