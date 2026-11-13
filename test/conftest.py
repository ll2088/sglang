"""Pytest bootstrap for the lightweight import smoke tests.

This conftest runs before any test module is collected.  It makes the source
tree importable (``python/`` on ``sys.path``) and installs minimal stubs for
heavy third-party dependencies that ``sglang/__init__.py`` transitively needs,
so the import smoke tests can run in a bare environment where the ``sglang``
distribution and its runtime dependency tree were never installed.

Real installations are never shadowed: a stub is only created when the module
is neither already imported nor resolvable via ``importlib.util.find_spec``.
"""

import importlib.util
import os
import sys
import types

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")

# Heavy third-party modules that ``sglang/__init__.py`` (transitively) needs
# but that the smoke tests must not require to be installed.
_HEAVY_DEPENDENCY_STUBS = (
    "torch",
    "transformers",
    "flashinfer",
    "numpy",
    "requests",
    "pydantic",
    "pybase64",
    "IPython",
    "tqdm",
)


def _ensure_source_on_path() -> None:
    """Make ``python/sglang`` importable from the source tree."""
    if os.path.isdir(os.path.join(_PYTHON_DIR, "sglang")):
        if _PYTHON_DIR not in sys.path:
            sys.path.insert(0, _PYTHON_DIR)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)


def _install_lightweight_dependency_stubs() -> None:
    """Install minimal stubs for genuinely missing heavy dependencies."""
    for name in _HEAVY_DEPENDENCY_STUBS:
        if name in sys.modules:
            continue
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            continue

        stub = types.ModuleType(name)
        stub.__path__ = []  # type: ignore[attr-defined]
        stub.__version__ = "0.0.0+stub"  # type: ignore[attr-defined]

        class _StubAny:
            def __init__(self, *args, **kwargs):
                pass

            def __getattr__(self, item):
                return _StubAny()

            def __call__(self, *args, **kwargs):
                return _StubAny()

        def _stub_getattr(item):
            if item.startswith("__") and item.endswith("__"):
                raise AttributeError(item)
            return _StubAny()

        stub.__getattr__ = _stub_getattr  # type: ignore[attr-defined]
        sys.modules[name] = stub


_ensure_source_on_path()
_install_lightweight_dependency_stubs()
