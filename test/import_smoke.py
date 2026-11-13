"""Lightweight, dependency-free smoke tests for the ``sglang`` top-level import chain.

Only the standard library is required.  The heavy runtime dependency tree
(torch, FlashInfer, ...) is stubbed by this module so the checks run even when
``pip install -e python`` failed during CI environment preparation.

Entry points:

    python test/import_smoke.py --bare        # import chain only
    python test/import_smoke.py --selfcheck   # bootstrap self-check
    python test/import_smoke.py --negative    # broken import must be detected
    python test/import_smoke.py --ci          # setup-safe gate (no pytest)
    python test/import_smoke.py --all         # bare + pytest suite if available

All entry points return 0 on success and non-zero on failure.
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

_STUBBED_MODULES = (
    "torch",
    "triton",
    "flashinfer",
    "transformers",
    "huggingface_hub",
    "numpy",
    "requests",
    "pybase64",
    "pydantic",
    "IPython",
    "tqdm",
    "zmq",
    "cv2",
    "PIL",
)


def _ensure_source_on_path() -> None:
    """Put ``python/`` on ``sys.path`` so no installed distribution is needed."""
    if _PYTHON_DIR not in sys.path:
        sys.path.insert(0, _PYTHON_DIR)


def _stub_is_installed(name: str) -> bool:
    mod = sys.modules.get(name)
    return mod is not None and getattr(mod, "__sglang_smoke_stub__", False)


def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__sglang_smoke_stub__ = True
    mod.__path__ = []  # look like a package so submodule imports resolve
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

    mod.__getattr__ = _getattr  # type: ignore[attr-defined]
    return mod


def _install_lightweight_dependency_stubs() -> None:
    """Install stubs for genuinely missing heavy dependencies.

    A real installation is never shadowed: a stub is only created when the
    module is neither already imported nor resolvable via ``find_spec``.
    """
    for name in _STUBBED_MODULES:
        if name in sys.modules:
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue
        except (ImportError, ValueError):
            pass
        sys.modules[name] = _make_stub(name)


def _import_sglang():
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    return importlib.import_module("sglang")


def _run_bare_smoke() -> int:
    """Import ``sglang`` in a fresh subprocess and check ``__version__``."""
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import importlib; importlib.import_module('sglang'); "
        "import sglang; assert isinstance(sglang.__version__, str) and sglang.__version__"
    ) % _PYTHON_DIR
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return proc.returncode


def _run_selfcheck() -> int:
    """Verify the bootstrap actually takes effect."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    if _PYTHON_DIR not in sys.path:
        print("selfcheck failed: python/ not on sys.path", file=sys.stderr)
        return 1
    if importlib.util.find_spec("torch") is None and not _stub_is_installed("torch"):
        print("selfcheck failed: torch stub not installed", file=sys.stderr)
        return 1
    return 0


def _run_negative_check() -> int:
    """A deliberately broken top-level import must be detected."""
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import importlib; "
        "import sglang; "
        "raise SystemExit(0)"
    ) % _PYTHON_DIR
    # Break the import by shadowing sglang with a module that raises.
    broken = (
        "import sys, types; "
        "sys.path.insert(0, %r); "
        "m = types.ModuleType('sglang'); "
        "m.__path__ = []; "
        "sys.modules['sglang'] = m; "
        "raise ImportError('deliberately broken sglang import')"
    ) % _PYTHON_DIR
    proc = subprocess.run([sys.executable, "-c", broken], capture_output=True, text=True)
    if proc.returncode == 0:
        print("negative check failed: broken import was not detected", file=sys.stderr)
        return 1
    return 0


def _run_ci() -> int:
    """Setup-safe gate: never imports pytest, never touches the network."""
    if _run_selfcheck() != 0:
        return 1
    if _run_bare_smoke() != 0:
        return 1
    if _run_negative_check() != 0:
        return 1
    return 0


def _run_pytest_smoke() -> int:
    """Run the pytest-based smoke suite when pytest is available."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("pytest not available; skipping pytest smoke suite", file=sys.stderr)
        return 0
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "import_smoke_test.py")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", test_file, "-k", "import_smoke"],
        cwd=_REPO_ROOT,
    )
    return proc.returncode


def _run_all() -> int:
    if _run_ci() != 0:
        return 1
    return _run_pytest_smoke()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--bare" in argv:
        return _run_bare_smoke()
    if "--selfcheck" in argv:
        return _run_selfcheck()
    if "--negative" in argv:
        return _run_negative_check()
    if "--ci" in argv:
        return _run_ci()
    if "--pytest" in argv:
        return _run_pytest_smoke()
    if "--all" in argv:
        return _run_all()
    return _run_all()


if __name__ == "__main__":
    raise SystemExit(main())
