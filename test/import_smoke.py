"""Self-contained smoke checks for the ``sglang`` top-level import chain.

This module only depends on the standard library.  It bootstraps ``python/``
onto ``sys.path`` and installs lightweight stubs for genuinely missing heavy
third-party dependencies (torch, transformers, ...) so that the top-level
import chain can be exercised in a bare environment where the ``sglang``
distribution and its runtime dependency tree were never installed.

Entry points (all runnable without pytest and without any third-party
package installed):

    python test/import_smoke.py --bare
    python test/import_smoke.py --selfcheck
    python test/import_smoke.py --negative

Exit code 0 means success; any non-zero exit code means the import chain is
broken and the traceback is printed to stderr.
"""

from __future__ import annotations

import argparse
import importlib
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


def _stub_is_installed(name: str) -> bool:
    mod = sys.modules.get(name)
    return bool(getattr(mod, "__sglang_smoke_stub__", False))


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


def _import_sglang():
    """Import ``sglang`` after bootstrapping the source tree and stubs."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    return importlib.import_module("sglang")


def _run_bare_smoke() -> int:
    """Import ``sglang`` and assert the public API resolves.

    Runs in a subprocess so that a partially-initialised ``sglang`` package
    (or a stub installed by an earlier bootstrap) can never mask a real
    import regression in this interpreter.
    """
    import subprocess

    code = (
        "import importlib, sys;"
        "sys.path.insert(0, %r);"
        "import import_smoke as _s;"
        "_s._ensure_source_on_path();"
        "_s._install_lightweight_dependency_stubs();"
        "sglang = importlib.import_module('sglang');"
        "assert sglang.__version__;"
        "assert isinstance(sglang.__version__, str);"
        "[getattr(sglang, n) for n in sglang.__all__];"
        "[importlib.import_module(m) for m in "
        "('sglang._platform_stubs', 'sglang.version', 'sglang.utils')]"
    ) % (os.path.dirname(os.path.abspath(__file__)),)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(os.path.abspath(__file__)), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        return 1
    print("import_smoke: ok")
    return 0


def _run_selfcheck() -> int:
    """Verify the bootstrap actually took effect in this interpreter.

    The check is deliberately tolerant of a real installation: when a heavy
    dependency is genuinely importable, the stub is intentionally not used
    and that is a pass, not a failure.  Only the case where the dependency is
    missing *and* no stub was installed is a bootstrap failure.
    """
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()

    for name in _HEAVY_DEPENDENCY_STUBS:
        if _stub_is_installed(name):
            continue
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            # Real installation present; the stub is intentionally skipped.
            continue
        print(
            f"import_smoke selfcheck: {name} is missing and was not stubbed",
            file=sys.stderr,
        )
        return 1

    print("import_smoke selfcheck: ok")
    return 0


def _run_negative_check() -> int:
    """Run a subprocess that deliberately breaks ``sglang`` and must fail.

    The check runs in a subprocess so the deliberately broken import never
    pollutes this interpreter's ``sys.modules``.  It returns 0 when the
    negative case is correctly detected (i.e. the subprocess fails).

    The broken package is placed on ``PYTHONPATH`` *ahead* of the source
    tree so it shadows the real ``sglang`` package regardless of whether the
    distribution is installed; the subprocess is also run with ``-S``-free
    defaults so the original error text is preserved.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        broken_pkg = os.path.join(tmpdir, "sglang")
        os.makedirs(broken_pkg, exist_ok=True)
        with open(os.path.join(broken_pkg, "__init__.py"), "w") as f:
            f.write("raise ImportError('deliberately broken sglang import')\n")

        env = dict(os.environ)
        # Prepend the broken package so it shadows both the source tree and
        # any installed distribution.
        env["PYTHONPATH"] = tmpdir + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", "import sglang"],
            env=env,
            capture_output=True,
            text=True,
        )

    if proc.returncode == 0:
        print(
            "import_smoke negative: broken sglang import unexpectedly succeeded",
            file=sys.stderr,
        )
        return 1
    if "deliberately broken sglang import" not in (proc.stderr or ""):
        print(
            "import_smoke negative: failure did not surface the original error",
            file=sys.stderr,
        )
        print(proc.stderr, file=sys.stderr)
        return 1
    print("import_smoke negative: ok")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bare", action="store_true", help="run the bare smoke")
    parser.add_argument(
        "--selfcheck", action="store_true", help="verify the bootstrap"
    )
    parser.add_argument(
        "--negative", action="store_true", help="verify a broken import fails"
    )
    args = parser.parse_args(argv)

    if args.selfcheck:
        return _run_selfcheck()
    if args.negative:
        return _run_negative_check()
    # Default (and --bare) run the bare smoke.
    return _run_bare_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
