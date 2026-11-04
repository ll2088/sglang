#!/usr/bin/env python3
"""Pytest wrapper for the stdlib-only Dockerfile permission checks.

The check logic lives in ``checks.py`` so it can also run under a bare
``python3`` interpreter via ``run.py`` without pytest or sglang installed. This
module only re-exports the checks as ``test_*`` functions for pytest.

Usage:
    python3 -m pytest -q -c test/dockerfile_permissions/pytest.ini \
        test/dockerfile_permissions/
"""

import os
import sys

# Make the sibling checks.py importable regardless of pytest's import mode or
# the caller's CWD. This keeps the checks stdlib-only and avoids importing the
# repository-root sglang package during collection.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from checks import (  # noqa: F401  (re-exported for pytest collection)
        check_devcontainer_workspace_not_world_writable as test_devcontainer_workspace_not_world_writable,
        check_minimal_permission_statements as test_minimal_permission_statements,
        check_no_world_writable_permissions as test_no_world_writable_permissions,
        check_readme_keyword_hit_threshold as test_readme_keyword_hit_threshold,
        check_readme_non_root_cache_guidance as test_readme_non_root_cache_guidance,
    )
except ImportError:  # pragma: no cover - only hit if checks.py is missing
    # Re-raise with a clear message so CI logs point at the real problem
    # instead of a generic collection error.
    raise ImportError(
        "test/dockerfile_permissions/checks.py is missing or not importable; "
        "the Dockerfile permission checks must remain stdlib-only"
    )


def test_all_checks_pass():
    """Run every check so a single pytest invocation covers the full suite."""
    from checks import ALL_CHECKS

    for check in ALL_CHECKS:
        check()

