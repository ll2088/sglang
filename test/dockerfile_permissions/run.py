#!/usr/bin/env python3
"""Standalone runner for the stdlib-only Dockerfile permission checks.

This entry point lets CI run the permission and documentation checks with a
bare ``python3`` interpreter, without installing sglang or pytest. It imports
the checks from ``test_dockerfile_permissions.py`` in the same directory and
reports a non-zero exit code on failure.

Usage:
    python3 test/dockerfile_permissions/run.py
"""

import sys
from pathlib import Path

# Make the sibling test module importable regardless of the caller's CWD.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from test_dockerfile_permissions import (  # noqa: E402
    CHECKS,
    missing_paths,
    readme_keyword_hit_count,
)


def main() -> int:
    missing = missing_paths()
    if missing:
        print("missing paths:")
        for item in missing:
            print(f"  {item}")
        return 1

    failures = 0
    for check in CHECKS:
        try:
            check()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {check.__name__}: {exc}")
        else:
            print(f"PASS {check.__name__}")

    print(f"readme_keyword_hits_total={readme_keyword_hit_count()}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
