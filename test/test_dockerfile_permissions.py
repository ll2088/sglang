#!/usr/bin/env python3
"""Compatibility entry point for the Dockerfile permission checks.

This file exists so CI commands that reference
``test/test_dockerfile_permissions.py`` keep working. It delegates to the
stdlib-only checks under ``test/dockerfile_permissions/`` and can be run either
as a script or via pytest.

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

from checks import (  # noqa: E402,F401  (re-exported for pytest collection)
    ALL_CHECKS,
    check_devcontainer_workspace_not_world_writable as test_devcontainer_workspace_not_world_writable,
    check_minimal_permission_statements as test_minimal_permission_statements,
    check_no_world_writable_permissions as test_no_world_writable_permissions,
    check_readme_keyword_hit_threshold as test_readme_keyword_hit_threshold,
    check_readme_non_root_cache_guidance as test_readme_non_root_cache_guidance,
)


def test_all_checks_pass():
    for check in ALL_CHECKS:
        check()


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
