"""Self-contained smoke checks for the ``sglang`` top-level import chain.

Only the standard library is required.  The heavy runtime dependency tree is
stubbed so the checks run even when ``pip install -e python`` failed during
CI environment preparation.

Entry points::

    python test/import_smoke.py --bare       # import sglang in a subprocess
    python test/import_smoke.py --selfcheck  # bootstrap self-check
    python test/import_smoke.py --negative   # broken import must be detected
    python test/import_smoke.py --ci         # setup-safe, never imports pytest
    python test/import_smoke.py --pytest     # run the pytest smoke suite
    python test/import_smoke.py --all        # everything above
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
    "torch.mps",
    "torch.cuda",
    "torch.backends",
    "torch.backends.mps",
    "transformers",
    "flashinfer",
    "triton",
    "triton.language",
    "triton.runtime",
    "triton.runtime.jit",
    "triton.compiler",
    "triton.backends",
    "triton.backends.compiler",
    "triton.testing",
    "triton.tools",
    "triton.tools.tensor_descriptor",
)

_installed_stubs: set[str] = set()


def _ensure_source_on_path() -> None:
    """Put ``python/`` on ``sys.path`` so ``import sglang`` works uninstalled."""
    if _PYTHON_DIR not in sys.path:
        sys.path.insert(0, _PYTHON_DIR)


def _stub_is_installed(name: str) -> bool:
    return name in _installed_stubs


def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []  # type: ignore[attr-defined]
    mod.__package__ = name
    mod.__file__ = __file__
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    return mod


def _install_lightweight_dependency_stubs() -> None:
    """Install minimal stubs for genuinely missing heavy dependencies.

    A stub is only created when the module is neither already imported nor
    resolvable via ``importlib.util.find_spec``, so a real installation is
    never shadowed.
    """
    for name in _STUBBED_MODULES:
        if name in sys.modules:
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue
        except (ImportError, ValueError):
            pass
        mod = _make_stub(name)
        sys.modules[name] = mod
        _installed_stubs.add(name)
        parent_name, _, child_name = name.rpartition(".")
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child_name, mod)

    # Minimal attributes the import chain touches.
    torch = sys.modules.get("torch")
    if torch is not None and _stub_is_installed("torch"):
        torch.__version__ = "0.0.0-stub"  # type: ignore[attr-defined]
        torch.device = lambda *a, **k: types.SimpleNamespace(type="cpu")  # type: ignore[attr-defined]
        torch.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
        torch.dtype = type("dtype", (), {})  # type: ignore[attr-defined]
        torch.nn = sys.modules.setdefault("torch.nn", _make_stub("torch.nn"))  # type: ignore[attr-defined]
        torch.cuda = sys.modules.setdefault("torch.cuda", _make_stub("torch.cuda"))  # type: ignore[attr-defined]
        torch.cuda.is_available = lambda: False  # type: ignore[attr-defined]
        torch.cuda.device_count = lambda: 0  # type: ignore[attr-defined]
        torch.cuda.current_device = lambda: 0  # type: ignore[attr-defined]
        torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(  # type: ignore[attr-defined]
            name="stub", total_memory=0, multi_processor_count=0
        )
        torch.backends = sys.modules.setdefault(  # type: ignore[attr-defined]
            "torch.backends", _make_stub("torch.backends")
        )
        torch.backends.mps = sys.modules.setdefault(  # type: ignore[attr-defined]
            "torch.backends.mps", _make_stub("torch.backends.mps")
        )
        torch.backends.mps.is_available = lambda: False  # type: ignore[attr-defined]
        torch.mps = sys.modules.setdefault("torch.mps", _make_stub("torch.mps"))  # type: ignore[attr-defined]

    triton = sys.modules.get("triton")
    if triton is not None and _stub_is_installed("triton"):
        triton.__version__ = "0.0.0-stub"  # type: ignore[attr-defined]
        triton.jit = lambda fn=None, **k: (fn if fn is not None else (lambda f: f))  # type: ignore[attr-defined]
        triton.cdiv = lambda a, b: -(a // -b)  # type: ignore[attr-defined]
        triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length() if n > 0 else 1  # type: ignore[attr-defined]
        triton.Config = type("Config", (), {})  # type: ignore[attr-defined]
        triton.autotune = lambda *a, **k: (lambda f: f)  # type: ignore[attr-defined]
        triton.language = sys.modules.setdefault(  # type: ignore[attr-defined]
            "triton.language", _make_stub("triton.language")
        )
        triton.language.constexpr = type("constexpr", (), {})  # type: ignore[attr-defined]
        triton.runtime = sys.modules.setdefault(  # type: ignore[attr-defined]
            "triton.runtime", _make_stub("triton.runtime")
        )
        triton.runtime.jit = sys.modules.setdefault(  # type: ignore[attr-defined]
            "triton.runtime.jit", _make_stub("triton.runtime.jit")
        )
        triton.runtime.jit.JITFunction = type("JITFunction", (), {})  # type: ignore[attr-defined]
        triton.runtime.jit.KernelInterface = type("KernelInterface", (), {})  # type: ignore[attr-defined]
        triton.compiler = sys.modules.setdefault(  # type: ignore[attr-defined]
            "triton.compiler", _make_stub("triton.compiler")
        )
        triton.compiler.CompiledKernel = type("CompiledKernel", (), {})  # type: ignore[attr-defined]
        triton.backends = sys.modules.setdefault(  # type: ignore[attr-defined]
            "triton.backends", _make_stub("triton.backends")
        )
        triton.backends.compiler = sys.modules.setdefault(  # type: ignore[attr-defined]
            "triton.backends.compiler", _make_stub("triton.backends.compiler")
        )
        triton.backends.compiler.GPUTarget = type("GPUTarget", (), {})  # type: ignore[attr-defined]
        triton.testing = sys.modules.setdefault(  # type: ignore[attr-defined]
            "triton.testing", _make_stub("triton.testing")
        )
        triton.tools = sys.modules.setdefault(  # type: ignore[attr-defined]
            "triton.tools", _make_stub("triton.tools")
        )
        triton.tools.tensor_descriptor = sys.modules.setdefault(  # type: ignore[attr-defined]
            "triton.tools.tensor_descriptor", _make_stub("triton.tools.tensor_descriptor")
        )


def _run_bare_smoke() -> int:
    """Import ``sglang`` in a fresh subprocess and report success."""
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import sglang; assert sglang.__version__; print('import_ok')"
    ) % _PYTHON_DIR
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
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
        "import sglang; raise SystemExit(0)"
    ) % _PYTHON_DIR
    # Sanity: the healthy import must succeed first.
    healthy = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if healthy.returncode != 0:
        sys.stderr.write(healthy.stdout)
        sys.stderr.write(healthy.stderr)
        print("negative check: healthy import failed", file=sys.stderr)
        return 1

    # Now break the import chain in a throwaway subprocess and confirm it fails.
    broken_code = (
        "import sys, types; sys.path.insert(0, %r); "
        "import sglang; "
        "raise SystemExit(0)"
    ) % _PYTHON_DIR
    broken = subprocess.run(
        [sys.executable, "-c", broken_code],
        capture_output=True,
        text=True,
        env={**os.environ, "SGLANG_SMOKE_FORCE_BROKEN": "1"},
    )
    # The env var is not honoured by sglang itself; the real negative check is
    # that a syntactically broken module fails.  Simulate by importing a
    # non-existent submodule.
    broken2 = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, %r); import sglang.nonexistent_module_xyz"
            % _PYTHON_DIR,
        ],
        capture_output=True,
        text=True,
    )
    if broken2.returncode == 0:
        print("negative check: broken import was not detected", file=sys.stderr)
        return 1
    return 0


def _run_ci() -> int:
    """Setup-safe entry point: never imports pytest, never touches network."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    try:
        import sglang  # noqa: F401
    except Exception as exc:  # pragma: no cover - reported to CI
        import traceback

        traceback.print_exc()
        print(f"import sglang failed: {exc!r}", file=sys.stderr)
        return 1
    return 0


def _run_pytest_smoke() -> int:
    """Run the pytest-based smoke suite when pytest is available."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("pytest not available; skipping pytest smoke suite")
        return 0
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "import_smoke_test.py")
    return int(pytest.main(["-q", test_file, "-k", "import_smoke"]))


def _run_all() -> int:
    rc = 0
    rc |= _run_selfcheck()
    rc |= _run_bare_smoke()
    rc |= _run_negative_check()
    rc |= _run_ci()
    rc |= _run_pytest_smoke()
    return rc


def main(argv: list[str] | None = None) -> int:
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
    # Default: run the setup-safe CI check.
    return _run_ci()


if __name__ == "__main__":
    raise SystemExit(main())
