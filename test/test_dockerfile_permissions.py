#!/usr/bin/env python3
"""Compatibility entry point for the stdlib-only Dockerfile permission checks.

This module lets the checks be invoked with the paths used by the validation
plan (``python3 test/test_dockerfile_permissions.py`` and
``python3 -m pytest -q test/test_dockerfile_permissions.py``) while the actual
implementation lives in ``test/dockerfile_permissions/``.

It imports only the standard library and the sibling ``checks`` module, so it
runs without installing sglang or any third-party package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CHECKS_DIR = _HERE / "dockerfile_permissions"
if str(_CHECKS_DIR) not in sys.path:
    sys.path.insert(0, str(_CHECKS_DIR))

from checks import (  # noqa: E402,F401  (re-exported for pytest collection)
    ALL_CHECKS,
    check_devcontainer_workspace_not_world_writable as test_devcontainer_workspace_not_world_writable,
    check_minimal_permission_statements as test_minimal_permission_statements,
    check_no_world_writable_permissions as test_no_world_writable_permissions,
    check_readme_keyword_hit_threshold as test_readme_keyword_hit_threshold,
    check_readme_non_root_cache_guidance as test_readme_non_root_cache_guidance,
    run_all,
)


def test_all_checks_pass() -> None:
    """Run every check so a single pytest invocation covers the full suite."""
    for check in ALL_CHECKS:
        check()


if __name__ == "__main__":
    raise SystemExit(run_all())
