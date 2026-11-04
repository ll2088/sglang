#!/usr/bin/env python3
"""Standalone runner for the stdlib-only Dockerfile permission checks.

Usage:
    python3 test/dockerfile_permissions/run.py

Exits 0 when all checks pass, 1 otherwise. Requires only the Python standard
library so CI does not need to install sglang or pytest.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import checks  # noqa: E402  (path set above)

CHECKS = [
    checks.check_no_world_writable_permissions,
    checks.check_minimal_permission_statements,
    checks.check_devcontainer_workspace_not_world_writable,
    checks.check_readme_non_root_cache_guidance,
    checks.check_readme_keyword_hit_threshold,
]


def main():
    failures = []
    for check in CHECKS:
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
