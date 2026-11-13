"""Self-contained smoke tests for the ``sglang`` top-level import chain.

These tests only depend on the standard library and pytest.  They bootstrap
``python/`` onto ``sys.path`` and install lightweight stubs for genuinely
missing heavy third-party dependencies (torch, transformers, ...) so that
``import sglang`` can be exercised even when the ``sglang`` distribution and
its runtime dependency tree were never installed (e.g. when CI environment
preparation fails while resolving heavy wheels).

Run with::

    python -m pytest -q test/import_smoke.py -k import_smoke

or, in a bare environment without pytest::

    python test/import_smoke.py --bare
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
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
    """Install minimal stubs for genuinely missing heavy dependencies.

    A stub is only created when the module is neither already imported nor
    resolvable via ``importlib.util.find_spec``; real installations are never
    shadowed.
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


def _bootstrap() -> None:
    """Idempotent bootstrap used by both pytest and the ``--bare`` entry."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()


# Bootstrap at import time so that pytest collection of this module (and of
# test/run_suite.py's import_smoke tests) sees a usable ``sglang`` package
# without requiring ``pip install -e python`` to have succeeded.
_bootstrap()


def _run_negative_check() -> int:
    """Run a subprocess that deliberately breaks ``sglang`` and assert failure.

    Returns the subprocess exit code.  A non-zero exit code means the smoke
    check correctly detected the broken import; zero means the check is not
    actually exercising the import chain and the caller should fail.
    """
    script = (
        "import importlib, sys\n"
        "sys.path.insert(0, {python_dir!r})\n"
        "sys.path.insert(0, {repo_root!r})\n"
        "import import_smoke\n"
        "import_smoke._install_lightweight_dependency_stubs()\n"
        "import sglang\n"
        "raise SystemExit(0)\n"
    ).format(python_dir=_PYTHON_DIR, repo_root=_REPO_ROOT)

    # Break the top-level package by shadowing it with a module that raises on
    # import.  We do this by injecting a fake ``sglang`` package directory on
    # sys.path ahead of the real one.
    broken_dir = os.path.join(_REPO_ROOT, "test", "_import_smoke_broken")
    os.makedirs(broken_dir, exist_ok=True)
    broken_pkg = os.path.join(broken_dir, "sglang")
    os.makedirs(broken_pkg, exist_ok=True)
    with open(os.path.join(broken_pkg, "__init__.py"), "w") as f:
        f.write("raise ImportError('deliberately broken sglang for smoke test')\n")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [broken_dir, _PYTHON_DIR, _REPO_ROOT, env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )
    # The subprocess must fail (non-zero) because the broken sglang shadows
    # the real one.  If it succeeds, the negative check is not effective.
    if proc.returncode == 0:
        return 1
    return 0


def test_import_smoke():
    """``import sglang`` must succeed and expose ``__version__``."""
    _bootstrap()
    sglang = importlib.import_module("sglang")
    assert sglang.__version__
    assert isinstance(sglang.__version__, str)

    for module_name in (
        "sglang._platform_stubs",
        "sglang.version",
        "sglang.utils",
    ):
        importlib.import_module(module_name)


def test_import_smoke_negative():
    """A broken top-level import must be detected (import-error dimension)."""
    assert _run_negative_check() == 0


def _main() -> int:
    if "--bare" in sys.argv:
        _bootstrap()
        importlib.import_module("sglang")
        importlib.import_module("sglang._platform_stubs")
        importlib.import_module("sglang.version")
        importlib.import_module("sglang.utils")
        assert _run_negative_check() == 0
        print("import_smoke: ok")
        return 0
    print("usage: python test/import_smoke.py --bare", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
