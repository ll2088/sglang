#!/usr/bin/env python3
"""Stdlib-only entrypoint for the Dockerfile permission/documentation checks.

CI invokes this file directly (``python3 test/test_dockerfile_permissions.py``)
and via pytest (``python3 -m pytest -q test/test_dockerfile_permissions.py``).
It must not import sglang or any third-party package at module scope so the
checks can run in a minimal environment without installing the full sglang
dependency tree.

The actual assertions live in ``test/dockerfile_permissions/checks.py``; this
module only re-exports them as ``test_*`` functions and provides a ``__main__``
runner for environments without pytest.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHECKS_DIR = os.path.join(_HERE, "dockerfile_permissions")
if _CHECKS_DIR not in sys.path:
    sys.path.insert(0, _CHECKS_DIR)

from checks import (  # noqa: E402  (re-exported for pytest collection)
    CHECKS,
    check_devcontainer_workspace_not_world_writable as test_devcontainer_workspace_not_world_writable,
    check_minimal_permission_statements as test_minimal_permission_statements,
    check_no_world_writable_permissions as test_no_world_writable_permissions,
    check_readme_keyword_hit_threshold as test_readme_keyword_hit_threshold,
    check_readme_non_root_cache_guidance as test_readme_non_root_cache_guidance,
    main as _checks_main,
)


def test_checks_pass():
    """All stdlib-only checks must pass on the current tree."""
    failures = []
    for check in CHECKS:
        failures.extend(check())
    assert not failures, "\n".join(failures)


def test_checks_module_is_stdlib_only():
    """The checks module must not import sglang or pytest."""
    import checks

    with open(checks.__file__, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert "import sglang" not in source
    assert "import pytest" not in source


if __name__ == "__main__":
    sys.exit(_checks_main())
