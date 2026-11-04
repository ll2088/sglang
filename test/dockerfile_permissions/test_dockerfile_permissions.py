#!/usr/bin/env python3
"""Pytest wrapper for the stdlib-only Dockerfile permission checks.

The check logic lives in ``checks.py`` so it can also run under a bare
``python3`` interpreter via ``run.py`` without pytest or sglang installed. This
module only re-exports the checks as ``test_*`` functions for pytest.

When pytest is not installed, running this file directly executes the checks
via ``checks.main()`` so CI can use a bare ``python3`` interpreter:

    python3 test/dockerfile_permissions/test_dockerfile_permissions.py

Usage with pytest:
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

from checks import (  # noqa: E402  (re-exported for pytest collection)
    check_devcontainer_workspace_not_world_writable as test_devcontainer_workspace_not_world_writable,
    check_minimal_permission_statements as test_minimal_permission_statements,
    check_no_world_writable_permissions as test_no_world_writable_permissions,
    check_readme_keyword_hit_threshold as test_readme_keyword_hit_threshold,
    check_readme_non_root_cache_guidance as test_readme_non_root_cache_guidance,
)


def test_checks_pass():
    """All stdlib-only checks must pass on the current tree."""
    from checks import CHECKS

    failures = []
    for check in CHECKS:
        failures.extend(check())
    assert not failures, "\n".join(failures)


def test_checks_module_is_stdlib_only():
    """The checks module must not import sglang or pytest."""
    import checks

    source = open(checks.__file__, "r", encoding="utf-8").read()
    assert "import sglang" not in source
    assert "import pytest" not in source


if __name__ == "__main__":
    # Allow `python3 test/dockerfile_permissions/test_dockerfile_permissions.py`
    # to run without pytest installed. Import lazily so pytest collection is
    # unaffected.
    from checks import main as _main

    sys.exit(_main())



