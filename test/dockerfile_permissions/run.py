#!/usr/bin/env python3
"""Standalone runner for the stdlib-only Dockerfile permission checks.

Runs without pytest or sglang installed and exits non-zero on any failure, so
CI can invoke it directly with a bare ``python3`` interpreter:

    python3 test/dockerfile_permissions/run.py
"""

import sys

from checks import CHECKS, missing_paths, readme_keyword_hit_count


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
