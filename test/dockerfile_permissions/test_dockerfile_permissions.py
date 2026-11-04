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
    # Fall back to the sibling checks.py by absolute path so pytest's import
    # mode or the caller's CWD cannot break collection. This keeps the checks
    # stdlib-only and avoids importing the repository-root sglang package.
    import importlib.util as _importlib_util

    _checks_path = os.path.join(_HERE, "checks.py")
    if not os.path.isfile(_checks_path):
        raise ImportError(
            "test/dockerfile_permissions/checks.py is missing or not importable; "
            "the Dockerfile permission checks must remain stdlib-only"
        )
    _spec = _importlib_util.spec_from_file_location("dockerfile_permissions_checks", _checks_path)
    _checks = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_checks)
    test_devcontainer_workspace_not_world_writable = _checks.check_devcontainer_workspace_not_world_writable
    test_minimal_permission_statements = _checks.check_minimal_permission_statements
    test_no_world_writable_permissions = _checks.check_no_world_writable_permissions
    test_readme_keyword_hit_threshold = _checks.check_readme_keyword_hit_threshold
    test_readme_non_root_cache_guidance = _checks.check_readme_non_root_cache_guidance



