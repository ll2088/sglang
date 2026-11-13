"""Self-contained smoke checks for the ``sglang`` top-level import chain.

Only the standard library is required.  The module bootstraps ``python/``
onto ``sys.path`` and installs lightweight stubs for genuinely missing heavy
runtime dependencies, so the checks run even when ``pip install -e python``
has not run -- or has failed during dependency resolution.

Entry points (all return an exit code, none raise on a healthy tree):

    python test/import_smoke.py --bare        # import sglang in a subprocess
    python test/import_smoke.py --selfcheck   # bootstrap actually took effect
    python test/import_smoke.py --negative    # a broken import is detected
    python test/import_smoke.py --ci          # bare + selfcheck, setup-safe
    python test/import_smoke.py --all         # everything, incl. pytest suite

The ``--ci`` entry point is the one CI should run *before* installing the
heavy runtime dependency tree: it never imports pytest and never touches the
network, so a failed environment-preparation step cannot mask a real import
regression.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
import types

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")

# Modules that ``sglang/__init__.py`` transitively needs but that are heavy or
# platform-specific.  A stub is only created when the real module is neither
# already imported nor resolvable, so a real installation is never shadowed.
_STUBBED_MODULES = (
    "torch",
    "torch.mps",
    "torch.cuda",
    "transformers",
    "flashinfer",
    "triton",
    "numpy",
    "requests",
    "pydantic",
    "pybase64",
    "IPython",
    "IPython.display",
    "tqdm",
)

_installed_stubs: set[str] = set()


def _ensure_source_on_path() -> None:
    """Put ``python/`` on ``sys.path`` so no installed distribution is needed."""
    if _PYTHON_DIR not in sys.path:
        sys.path.insert(0, _PYTHON_DIR)


def _stub_is_installed(name: str) -> bool:
    return name in _installed_stubs


def _make_stub(name: str) -> types.ModuleType:
    """Create a permissive stub module and register it in ``sys.modules``."""
    mod = types.ModuleType(name)
    mod.__path__ = []  # type: ignore[attr-defined]
    mod.__package__ = name
    mod.__file__ = __file__
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)

    def _getattr(attr: str):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        child_name = f"{name}.{attr}"
        if child_name in sys.modules:
            return sys.modules[child_name]
        if attr[:1].isupper():
            cls = type(attr, (), {"__module__": name})
            setattr(mod, attr, cls)
            return cls
        child = _make_stub(child_name)
        setattr(mod, attr, child)
        return child

    mod.__getattr__ = _getattr  # type: ignore[attr-defined]
    sys.modules[name] = mod
    _installed_stubs.add(name)
    return mod


def _install_lightweight_dependency_stubs() -> None:
    """Stub heavy deps that are genuinely missing; never shadow real ones."""
    for name in _STUBBED_MODULES:
        if name in sys.modules or _stub_is_installed(name):
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue
        except (ImportError, ValueError):
            pass
        _make_stub(name)


def _run_bare_smoke() -> int:
    """Import ``sglang`` in a fresh subprocess and report the outcome."""
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import sglang; "
        "assert sglang.__version__; "
        "print('import_ok', sglang.__version__)"
    ) % _PYTHON_DIR
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        print("import_smoke: bare import FAILED", file=sys.stderr)
        return 1
    print(proc.stdout.strip())
    return 0


def _run_selfcheck() -> int:
    """Verify the bare-environment bootstrap actually took effect."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    if importlib.util.find_spec("torch") is not None and not _stub_is_installed(
        "torch"
    ):
        print("import_smoke: real torch present; stub intentionally unused")
        return 0
    if not _stub_is_installed("torch"):
        print(
            "import_smoke: bootstrap self-check FAILED (torch stub missing)",
            file=sys.stderr,
        )
        return 1
    print("import_smoke: bootstrap self-check ok")
    return 0


def _run_negative_check() -> int:
    """A deliberately broken top-level import must be detected."""
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import sglang; "
        "raise SystemExit(0)"
    ) % _PYTHON_DIR
    env = dict(os.environ)
    env["SGLANG_IMPORT_SMOKE_FORCE_FAIL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
    )
    if proc.returncode == 0:
        print(
            "import_smoke: negative check FAILED (broken import not detected)",
            file=sys.stderr,
        )
        return 1
    print("import_smoke: negative check ok")
    return 0


def _run_ci() -> int:
    """Setup-safe entry point: run before the heavy install, never abort it.

    Returns 0 only when the bare import and the bootstrap self-check both
    pass.  Never imports pytest and never touches the network, so it is safe
    to run in a bare environment where ``pip install -e python`` failed.
    """
    rc = _run_bare_smoke()
    if rc != 0:
        return rc
    return _run_selfcheck()


def _run_pytest_smoke() -> int:
    """Run the pytest-based smoke suite when pytest is importable."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("import_smoke: pytest not installed; skipping pytest suite")
        return 0
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "import_smoke_test.py")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", test_file, "-k", "import_smoke"],
        cwd=_REPO_ROOT,
    )
    return proc.returncode


def _run_all() -> int:
    for step in (_run_bare_smoke, _run_selfcheck, _run_negative_check):
        rc = step()
        if rc != 0:
            return rc
    return _run_pytest_smoke()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bare", action="store_true", help="import sglang in a subprocess")
    parser.add_argument("--selfcheck", action="store_true", help="verify the bootstrap")
    parser.add_argument("--negative", action="store_true", help="verify a broken import is caught")
    parser.add_argument("--ci", action="store_true", help="setup-safe entry point")
    parser.add_argument("--pytest", action="store_true", help="run the pytest smoke suite")
    parser.add_argument("--all", action="store_true", help="run every check")
    args = parser.parse_args(argv)

    if args.all:
        return _run_all()
    if args.ci:
        return _run_ci()
    if args.bare:
        return _run_bare_smoke()
    if args.selfcheck:
        return _run_selfcheck()
    if args.negative:
        return _run_negative_check()
    if args.pytest:
        return _run_pytest_smoke()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
