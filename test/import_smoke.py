"""Self-contained smoke checks for the ``sglang`` top-level import chain.

This module only depends on the standard library.  It bootstraps ``python/``
onto ``sys.path`` and installs lightweight stubs for genuinely missing heavy
third-party dependencies (e.g. torch) so that the top-level import chain can
be exercised even when the ``sglang`` distribution and its runtime dependency
tree were never installed.

Entry points (all runnable without pytest):

    python test/import_smoke.py --bare
    python test/import_smoke.py --selfcheck
    python test/import_smoke.py --negative

A real installation is never shadowed: a stub is only created when the module
is neither already imported nor resolvable via ``importlib.util.find_spec``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")

# Heavy third-party modules that ``sglang/__init__.py`` (transitively) needs
# but that the smoke checks must not require to be installed.
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

# Modules whose import must succeed for the top-level import chain to be
# considered healthy.
_SMOKE_MODULES = (
    "sglang",
    "sglang._platform_stubs",
    "sglang.version",
    "sglang.utils",
)


def _ensure_source_on_path() -> None:
    """Make ``python/sglang`` importable from the source tree."""
    if os.path.isdir(os.path.join(_PYTHON_DIR, "sglang")):
        if _PYTHON_DIR not in sys.path:
            sys.path.insert(0, _PYTHON_DIR)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)


def _stub_is_installed(name: str) -> bool:
    """Return True when ``name`` is currently backed by a lightweight stub."""
    module = sys.modules.get(name)
    if module is None:
        return False
    return getattr(module, "__sglang_smoke_stub__", False) is True


def _install_lightweight_dependency_stubs() -> None:
    """Install minimal stubs for genuinely missing heavy dependencies.

    Idempotent and never shadows a real installation.
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
        stub.__sglang_smoke_stub__ = True  # type: ignore[attr-defined]

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


def _run_bare_smoke() -> int:
    """Run the smoke checks with the standard library only.

    Returns 0 on success, 1 on failure.  Never raises for an import error so
    that the caller can report a clean exit code.
    """
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()

    for module_name in _SMOKE_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - report and fail cleanly
            print(f"import_smoke: FAILED to import {module_name}: {exc!r}")
            return 1

    sglang = sys.modules["sglang"]
    version = getattr(sglang, "__version__", None)
    if not version or not isinstance(version, str):
        print("import_smoke: FAILED: sglang.__version__ is missing or not a str")
        return 1

    for name in getattr(sglang, "__all__", []):
        if not hasattr(sglang, name):
            print(f"import_smoke: FAILED: missing public symbol: {name}")
            return 1

    print(f"import_smoke: OK (sglang.__version__={version})")
    return 0


def _run_selfcheck() -> int:
    """Verify the bare-environment bootstrap actually takes effect."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()

    if importlib.util.find_spec("torch") is not None and not _stub_is_installed(
        "torch"
    ):
        print("import_smoke: selfcheck skipped (real torch installed)")
        return 0

    if not _stub_is_installed("torch"):
        print("import_smoke: selfcheck FAILED: torch stub not installed")
        return 1

    print("import_smoke: selfcheck OK")
    return 0


def _run_negative_check() -> int:
    """Verify a broken top-level import is detected.

    Runs in a subprocess so the deliberately broken import never pollutes this
    interpreter's ``sys.modules``.  Returns 0 when the negative case is
    correctly detected (i.e. the subprocess fails as expected), 1 otherwise.
    """
    import subprocess
    import tempfile

    script = (
        "import importlib, os, sys\n"
        "sys.path.insert(0, os.path.join(os.getcwd(), 'python'))\n"
        "import sglang\n"
        "raise SystemExit(0)\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build a shadow package that raises on import, placed ahead of the
        # real source tree on sys.path.
        shadow = os.path.join(tmpdir, "sglang")
        os.makedirs(shadow, exist_ok=True)
        with open(os.path.join(shadow, "__init__.py"), "w", encoding="utf-8") as f:
            f.write("raise ImportError('deliberately broken sglang import')\n")

        env = dict(os.environ)
        env["PYTHONPATH"] = tmpdir + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    if proc.returncode == 0:
        print("import_smoke: negative check FAILED: broken import was not detected")
        return 1

    combined = (proc.stdout or "") + (proc.stderr or "")
    if "deliberately broken sglang import" not in combined:
        print(
            "import_smoke: negative check FAILED: error message not surfaced\n"
            + combined
        )
        return 1

    print("import_smoke: negative check OK")
    return 0


if __name__ == "__main__":
    if "--bare" in sys.argv[1:]:
        raise SystemExit(_run_bare_smoke())
    if "--selfcheck" in sys.argv[1:]:
        raise SystemExit(_run_selfcheck())
    if "--negative" in sys.argv[1:]:
        raise SystemExit(_run_negative_check())
    raise SystemExit(_run_bare_smoke())
