#!/usr/bin/env python3
"""Stdlib-only Dockerfile permission and README guidance checks.

This file is the canonical CI entry point. It imports nothing outside the
Python standard library so it can run with a bare ``python3`` interpreter,
without installing sglang or pytest::

    python3 test/test_dockerfile_permissions.py

When pytest is available, the same checks are also exposed as ``test_*``
functions and can be run with::

    python3 -m pytest -q test/test_dockerfile_permissions.py

The check logic lives in ``test/dockerfile_permissions/checks.py`` so it can be
shared with the standalone runner and the pytest wrapper in that directory.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHECKS_DIR = os.path.join(_HERE, "dockerfile_permissions")
if _CHECKS_DIR not in sys.path:
    sys.path.insert(0, _CHECKS_DIR)

from checks import (  # noqa: E402
    CHECKS,
    check_devcontainer_workspace_not_world_writable,
    check_minimal_permission_statements,
    check_no_world_writable_permissions,
    check_readme_keyword_hit_threshold,
    check_readme_non_root_cache_guidance,
    main,
)


def test_no_world_writable_permissions():
    assert not check_no_world_writable_permissions()


def test_minimal_permission_statements():
    assert not check_minimal_permission_statements()


def test_devcontainer_workspace_not_world_writable():
    assert not check_devcontainer_workspace_not_world_writable()


def test_readme_non_root_cache_guidance():
    assert not check_readme_non_root_cache_guidance()


def test_readme_keyword_hit_threshold():
    assert not check_readme_keyword_hit_threshold()


def test_all_checks_pass():
    failures = []
    for check in CHECKS:
        failures.extend(check())
    assert not failures, "\n".join(failures)


def test_checks_module_is_stdlib_only():
    import checks

    with open(checks.__file__, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert "import sglang" not in source
    assert "import pytest" not in source


if __name__ == "__main__":
    sys.exit(main())
