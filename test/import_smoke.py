"""Self-contained bootstrap + smoke checks for the ``sglang`` import chain.

This module only depends on the standard library.  It is imported by file
path from ``test/import_smoke_test.py`` and ``test/run_suite.py`` so the two
smoke entry points share one implementation and cannot drift apart.

It exists so the import-time / import-error / import-compat dimensions can be
observed even when the heavy ``sglang`` runtime dependency tree (torch,
py-spy, pybase64, pydantic, pyzmq, quack-kernels, regex, ...) cannot be
installed during CI environment preparation.  ``python/`` is injected onto
``sys.path`` and lightweight stubs are installed for any genuinely missing
heavy dependency, so the top-level import chain is exercised directly from
source.  A real installation is never shadowed.

Entry points (no pytest required):

    python test/import_smoke.py --bare
    python test/import_smoke.py --selfcheck
    python test/import_smoke.py --negative
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

# ---------------------------------------------------------------------------
# sys.path bootstrap
# ---------------------------------------------------------------------------


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_source_on_path() -> None:
    """Put ``python/`` on ``sys.path`` so ``import sglang`` works from source.

    Idempotent: repeated calls do not duplicate the entry.  A real installed
    distribution is never shadowed -- the source tree is only prepended when
    it is not already present.
    """
    python_dir = os.path.join(_repo_root(), "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)


# ---------------------------------------------------------------------------
# Lightweight dependency stubs
# ---------------------------------------------------------------------------

_STUBBED_MODULES = (
    "torch",
    "torch.mps",
    "torch.cuda",
    "torch.backends",
    "torch.backends.mps",
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


def _stub_is_installed(name: str) -> bool:
    """Return True when ``name`` is currently provided by our stub."""
    mod = sys.modules.get(name)
    return mod is not None and getattr(mod, "__import_smoke_stub__", False)


def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__import_smoke_stub__ = True  # type: ignore[attr-defined]
    mod.__path__ = []  # type: ignore[attr-defined]
    mod.__package__ = name
    mod.__file__ = __file__
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    return mod


def _install_lightweight_dependency_stubs() -> None:
    """Install minimal stubs for genuinely missing heavy dependencies.

    Only modules that are *not* importable are stubbed, so a real installation
    is never shadowed.  Idempotent.
    """
    # torch
    if importlib.util.find_spec("torch") is None and not _stub_is_installed("torch"):
        torch = _make_stub("torch")
        torch.__version__ = "0.0.0-import-smoke-stub"

        class _Device:
            def __init__(self, spec=None, type=None):  # noqa: A002
                self.type = type or (str(spec).split(":", 1)[0] if spec else "cpu")

            def __str__(self):
                return self.type

        def _device(spec=None, type=None):  # noqa: A002
            return _Device(spec, type)

        torch.device = _device
        torch.Tensor = type("Tensor", (), {})
        torch.dtype = type("dtype", (), {})
        torch.float16 = torch.dtype()
        torch.bfloat16 = torch.dtype()
        torch.float32 = torch.dtype()

        mps = _make_stub("torch.mps")
        mps.is_available = lambda: False
        mps.current_allocated_memory = lambda: 0
        mps.driver_allocated_memory = lambda: 0
        mps.recommended_max_memory = lambda: 0
        torch.mps = mps

        cuda = _make_stub("torch.cuda")
        cuda.is_available = lambda: False
        cuda.device_count = lambda: 0
        torch.cuda = cuda

        backends = _make_stub("torch.backends")
        backends_mps = _make_stub("torch.backends.mps")
        backends_mps.is_available = lambda: False
        backends.mps = backends_mps
        torch.backends = backends

        sys.modules["torch"] = torch
        sys.modules["torch.mps"] = mps
        sys.modules["torch.cuda"] = cuda
        sys.modules["torch.backends"] = backends
        sys.modules["torch.backends.mps"] = backends_mps

    # triton
    if importlib.util.find_spec("triton") is None and not _stub_is_installed(
        "triton"
    ):
        triton = _make_stub("triton")
        triton.__version__ = "0.0.0-import-smoke-stub"
        triton.cdiv = lambda a, b: -(a // -b)
        triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length() if n > 0 else 1

        class _Config:
            def __init__(self, kwargs=None, num_warps=4, num_stages=2, **extra):
                self.kwargs = kwargs or {}
                self.num_warps = num_warps
                self.num_stages = num_stages

        triton.Config = _Config

        tl = _make_stub("triton.language")

        class _constexpr:
            def __init__(self, value=None):
                self.value = value

        tl.constexpr = _constexpr
        triton.language = tl

        runtime = _make_stub("triton.runtime")
        jit_mod = _make_stub("triton.runtime.jit")
        jit_mod.JITFunction = type("JITFunction", (), {})
        jit_mod.KernelInterface = type("KernelInterface", (), {})
        runtime.jit = jit_mod
        triton.runtime = runtime

        compiler_root = _make_stub("triton.compiler")
        compiler_root.CompiledKernel = type("CompiledKernel", (), {})
        triton.compiler = compiler_root

        backends = _make_stub("triton.backends")
        backends_compiler = _make_stub("triton.backends.compiler")
        backends_compiler.GPUTarget = type("GPUTarget", (), {})
        backends.compiler = backends_compiler
        triton.backends = backends

        testing = _make_stub("triton.testing")
        triton.testing = testing

        tools = _make_stub("triton.tools")
        tools.tensor_descriptor = _make_stub("triton.tools.tensor_descriptor")
        triton.tools = tools

        for name, mod in (
            ("triton", triton),
            ("triton.language", tl),
            ("triton.runtime", runtime),
            ("triton.runtime.jit", jit_mod),
            ("triton.compiler", compiler_root),
            ("triton.backends", backends),
            ("triton.backends.compiler", backends_compiler),
            ("triton.testing", testing),
            ("triton.tools", tools),
            ("triton.tools.tensor_descriptor", tools.tensor_descriptor),
        ):
            sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Smoke checks (no pytest required)
# ---------------------------------------------------------------------------

_SMOKE_MODULES = (
    "sglang._platform_stubs",
    "sglang.version",
    "sglang.utils",
)


def _run_bare_smoke() -> int:
    """Import ``sglang`` and check ``__version__``; return a process exit code."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    try:
        sglang = importlib.import_module("sglang")
    except Exception as exc:  # noqa: BLE001 - report any import failure
        print(f"import_smoke: FAIL ({type(exc).__name__}: {exc})")
        return 1

    version = getattr(sglang, "__version__", None)
    if not version or not isinstance(version, str):
        print(f"import_smoke: FAIL (bad __version__: {version!r})")
        return 1

    print(f"import_smoke: OK (sglang.__version__={version})")
    return 0


def _run_selfcheck() -> int:
    """Run every smoke check without pytest; return a process exit code."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()

    failures = []

    try:
        sglang = importlib.import_module("sglang")
    except Exception as exc:  # noqa: BLE001 - report any import failure
        print(f"import_smoke: FAIL (sglang: {type(exc).__name__}: {exc})")
        return 1

    version = getattr(sglang, "__version__", None)
    if not version or not isinstance(version, str):
        failures.append(f"sglang.__version__ invalid: {version!r}")
    else:
        print(f"import_smoke: OK (sglang.__version__={version})")

    for module_name in _SMOKE_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - report any import failure
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
        else:
            print(f"import_smoke: OK ({module_name})")

    for name in getattr(sglang, "__all__", ()):
        if not hasattr(sglang, name):
            failures.append(f"sglang.__all__ missing symbol: {name}")
    if not failures:
        print("import_smoke: OK (sglang.__all__)")

    if failures:
        for failure in failures:
            print(f"import_smoke: FAIL ({failure})")
        return 1
    return 0


def _run_negative_check() -> int:
    """Break ``sglang/__init__.py`` in a subprocess and confirm it is caught."""
    import subprocess

    repo_root = _repo_root()
    init_path = os.path.join(repo_root, "python", "sglang", "__init__.py")
    if not os.path.isfile(init_path):
        print("import_smoke_negative: SKIP (sglang/__init__.py not found)")
        return 0

    with open(init_path, encoding="utf-8") as f:
        original = f.read()

    broken = "raise ImportError('import_smoke_negative_check')\n" + original
    try:
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(broken)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--bare"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
    finally:
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(original)

    if proc.returncode == 0:
        print(
            "import_smoke_negative: FAIL (broken import was not detected)\n"
            + proc.stdout
            + proc.stderr
        )
        return 1

    print("import_smoke_negative: OK (broken import detected)")
    return 0


if __name__ == "__main__":
    if "--bare" in sys.argv[1:]:
        raise SystemExit(_run_bare_smoke())
    if "--selfcheck" in sys.argv[1:]:
        raise SystemExit(_run_selfcheck())
    if "--negative" in sys.argv[1:]:
        raise SystemExit(_run_negative_check())
    raise SystemExit(_run_selfcheck())
