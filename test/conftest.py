"""Pytest bootstrap for the lightweight import smoke tests.

This conftest runs before pytest collects any test module.  It makes the
``sglang`` source tree importable and installs minimal stubs for the heavy
third-party dependencies of ``sglang`` so that the import smoke tests can run
in a bare environment (standard library + pytest only), without requiring
``pip install -e python`` to have succeeded.

A failed CI environment-preparation step (e.g. resolving py-spy / pybase64 /
pydantic / pyzmq / quack-kernels) must not mask a real import regression, so
the bootstrap is intentionally self-contained and never shadows a real
installation.
"""

import importlib.util
import os
import sys
import types


def _ensure_source_on_path() -> None:
    """Make ``python/sglang`` importable from the source tree."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_dir = os.path.join(repo_root, "python")
    if os.path.isdir(os.path.join(python_dir, "sglang")):
        if python_dir not in sys.path:
            sys.path.insert(0, python_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


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


def _install_lightweight_dependency_stubs() -> None:
    """Install minimal stubs for heavy third-party deps of ``sglang``.

    A stub is only created when the module is neither already imported nor
    resolvable via ``importlib.util.find_spec``, so a real installation is
    never shadowed.
    """
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

        def _stub_getattr(item, _name=name):
            if item.startswith("__") and item.endswith("__"):
                raise AttributeError(item)
            return _StubAny()

        stub.__getattr__ = _stub_getattr  # type: ignore[attr-defined]
        sys.modules[name] = stub


class _StubAny:
    """Permissive stand-in for any attribute of a stubbed dependency."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return _StubAny()

    def __call__(self, *args, **kwargs):
        return _StubAny()


_ensure_source_on_path()
_install_lightweight_dependency_stubs()
