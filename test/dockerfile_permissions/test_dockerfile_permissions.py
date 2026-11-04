#!/usr/bin/env python3
"""Pytest wrapper for the stdlib-only Dockerfile permission checks.

The check logic lives in ``checks.py`` so it can also run under a bare
``python3`` interpreter via ``run.py`` without pytest or sglang installed. This
module only re-exports the checks as ``test_*`` functions for pytest.

Usage:
    python3 -m pytest -q -c test/dockerfile_permissions/pytest.ini \
        test/dockerfile_permissions/
"""

from checks import (  # noqa: F401  (re-exported for pytest collection)
    check_devcontainer_workspace_not_world_writable as test_devcontainer_workspace_not_world_writable,
    check_minimal_permission_statements as test_minimal_permission_statements,
    check_no_world_writable_permissions as test_no_world_writable_permissions,
    check_readme_keyword_hit_threshold as test_readme_keyword_hit_threshold,
    check_readme_non_root_cache_guidance as test_readme_non_root_cache_guidance,
)

