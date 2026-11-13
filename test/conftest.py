"""Pytest bootstrap for the lightweight import smoke tests.

Runs before any test module is collected.  Makes the source tree importable
(``python/`` on ``sys.path``) and installs minimal stubs for heavy third-party
dependencies that ``sglang/__init__.py`` transitively needs, so the import
smoke tests can run in a bare environment where the ``sglang`` distribution
and its runtime dependency tree were never installed.

Real installations are never shadowed: a stub is only created when the module
is neither already imported nor resolvable via ``importlib.util.find_spec``.
"""

import importlib.util
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")


def _load_smoke_module():
    """Load ``test/import_smoke.py`` by file path (no package required)."""
    _smoke_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "import_smoke.py"
    )
    _spec = importlib.util.spec_from_file_location("import_smoke", _smoke_path)
    assert _spec is not None and _spec.loader is not None
    _smoke = importlib.util.module_from_spec(_spec)
    sys.modules.setdefault("import_smoke", _smoke)
    _spec.loader.exec_module(_smoke)
    return _smoke


_smoke = _load_smoke_module()
_ensure_source_on_path = _smoke._ensure_source_on_path
_install_lightweight_dependency_stubs = _smoke._install_lightweight_dependency_stubs

_ensure_source_on_path()
_install_lightweight_dependency_stubs()
