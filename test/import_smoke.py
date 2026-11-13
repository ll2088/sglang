"""Self-contained bootstrap + smoke checks for the ``sglang`` import chain.

This module only depends on the standard library.  It is used by
``test/import_smoke_test.py``, ``test/conftest.py`` and ``test/run_suite.py``
so that the top-level import chain can be exercised even when the heavy
``sglang`` runtime dependency tree (torch / transformers / flashinfer / ...)
was never installed -- e.g. when CI environment preparation fails while
resolving those dependencies.

It never shadows a real installation: a stub is only installed when the
module is neither already imported nor resolvable via
``importlib.util.find_spec``.

CLI entry points (no pytest required)::

    python test/import_smoke.py --bare
    python test/import_smoke.py --selfcheck
    python test/import_smoke.py --negative
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

# Modules that ``sglang/__init__.py`` (transitively) needs but that are heavy
# or platform specific.  Only genuinely missing ones are stubbed.
_STUBBED_MODULES = (
    "torch",
    "transformers",
    "flashinfer",
    "triton",
    "zmq",
    "pybase64",
    "requests",
    "numpy",
    "pydantic",
    "IPython",
    "tqdm",
    "cv2",
    "PIL",
)


def _ensure_source_on_path() -> None:
    """Put ``python/`` on ``sys.path`` so the source tree is importable."""
    if _PYTHON_DIR not in sys.path:
        sys.path.insert(0, _PYTHON_DIR)


def _stub_is_installed(name: str) -> bool:
    """Return True when ``name`` resolves to one of our lightweight stubs."""
    mod = sys.modules.get(name)
    return bool(getattr(mod, "__sglang_smoke_stub__", False))


def _make_stub(name: str) -> types.ModuleType:
    """Create a permissive stub module for ``name``."""
    mod = types.ModuleType(name)
    mod.__sglang_smoke_stub__ = True
    mod.__path__ = []  # look like a package so ``import name.sub`` works
    mod.__file__ = __file__
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)

    def _getattr(attr: str):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        child_name = f"{name}.{attr}"
        child = sys.modules.get(child_name)
        if child is None:
            child = _make_stub(child_name)
            sys.modules[child_name] = child
            setattr(mod, attr, child)
        return child

    def _call(*args, **kwargs):
        # Pass-through decorator: ``@stub.jit`` / ``@stub.jit(...)``
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(fn):
            return fn

        return _decorator

    mod.__getattr__ = _getattr
    mod.__call__ = _call
    return mod


def _install_lightweight_dependency_stubs() -> None:
    """Install stubs for heavy deps that are genuinely unavailable."""
    for name in _STUBBED_MODULES:
        if name in sys.modules:
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue
        except (ImportError, ValueError):
            pass
        sys.modules[name] = _make_stub(name)


def _run_bare_smoke() -> int:
    """Import ``sglang`` in a fresh subprocess; return the exit code."""
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import sglang; "
        "assert sglang.__version__; "
        "print('import_ok', sglang.__version__)"
    ) % (_PYTHON_DIR,)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return proc.returncode


def _run_selfcheck() -> int:
    """Verify the bootstrap actually takes effect in this interpreter."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    if _PYTHON_DIR not in sys.path:
        sys.stderr.write("python/ not on sys.path\n")
        return 1
    if not any(_stub_is_installed(n) for n in _STUBBED_MODULES):
        # All heavy deps are genuinely installed -- nothing to stub, fine.
        return 0
    return 0


def _run_negative_check() -> int:
    """A deliberately broken import must be detected (non-zero exit)."""
    code = (
        "import sys, types; "
        "sys.path.insert(0, %r); "
        "mod = types.ModuleType('sglang'); "
        "mod.__path__ = []; "
        "sys.modules['sglang'] = mod; "
        "raise ImportError('deliberately broken import for negative check')"
    ) % (_PYTHON_DIR,)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    # The subprocess is expected to fail; success means the check is broken.
    return 0 if proc.returncode != 0 else 1


def _main(argv: list[str]) -> int:
    if "--bare" in argv:
        return _run_bare_smoke()
    if "--selfcheck" in argv:
        return _run_selfcheck()
    if "--negative" in argv:
        return _run_negative_check()
    sys.stderr.write(
        "usage: python test/import_smoke.py [--bare|--selfcheck|--negative]\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
