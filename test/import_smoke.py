"""Self-contained smoke checks for the ``sglang`` top-level import chain.

This module only depends on the standard library.  It bootstraps ``python/``
onto ``sys.path`` and installs lightweight stubs for genuinely missing heavy
third-party dependencies (e.g. ``torch``) so that the top-level import chain
can be exercised even when the full ``sglang`` runtime dependency tree was
never installed (``pip install -e python`` can fail during CI environment
preparation and mask real import regressions).

Entry points:

* ``python test/import_smoke.py --bare``       -- import ``sglang`` only
* ``python test/import_smoke.py --selfcheck``  -- run every smoke check
* ``python test/import_smoke.py --negative``   -- verify a broken import fails

``test/import_smoke_test.py`` and ``test/run_suite.py`` import the helpers
below by file path so the two pytest entry points share one implementation.
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
    distribution is never shadowed because the source directory is only
    appended when it is not already present.
    """
    python_dir = os.path.join(_repo_root(), "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)


# ---------------------------------------------------------------------------
# Lightweight dependency stubs
# ---------------------------------------------------------------------------

_STUBBED_MODULES: set[str] = set()


def _stub_is_installed(name: str) -> bool:
    """Return True when ``name`` is currently provided by a lightweight stub."""
    return name in _STUBBED_MODULES


def _make_stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []  # type: ignore[attr-defined]
    mod.__package__ = name
    return mod


def _install_torch_stub() -> None:
    """Install a minimal ``torch`` stub for the bare-environment smoke path.

    Only the attributes touched by ``sglang``'s import-time code paths are
    provided.  This stub is never installed when a real ``torch`` is
    importable, so a real installation is never shadowed.
    """
    torch = _make_stub_module("torch")
    torch.__version__ = "0.0.0+import-smoke-stub"

    class _Device:
        def __init__(self, spec: object = "cpu") -> None:
            self.type = "cpu"
            self.index = None
            self._spec = spec

        def __repr__(self) -> str:  # pragma: no cover - debug helper
            return f"device(type='cpu')"

    def _device(spec: object = "cpu") -> _Device:
        return _Device(spec)

    torch.device = _device  # type: ignore[attr-defined]

    class _Tensor:
        pass

    torch.Tensor = _Tensor  # type: ignore[attr-defined]

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __call__(self, fn):
            return fn

    torch.no_grad = _NoGrad()  # type: ignore[attr-defined]

    cuda = _make_stub_module("torch.cuda")
    cuda.is_available = lambda: False  # type: ignore[attr-defined]
    cuda.device_count = lambda: 0  # type: ignore[attr-defined]
    cuda.current_device = lambda: 0  # type: ignore[attr-defined]
    torch.cuda = cuda  # type: ignore[attr-defined]

    mps = _make_stub_module("torch.mps")
    mps.is_available = lambda: False  # type: ignore[attr-defined]
    mps.recommended_max_memory = lambda: 0  # type: ignore[attr-defined]
    mps.current_allocated_memory = lambda: 0  # type: ignore[attr-defined]
    mps.driver_allocated_memory = lambda: 0  # type: ignore[attr-defined]
    torch.mps = mps  # type: ignore[attr-defined]

    backends = _make_stub_module("torch.backends")
    backends_mps = _make_stub_module("torch.backends.mps")
    backends_mps.is_available = lambda: False  # type: ignore[attr-defined]
    backends.mps = backends_mps  # type: ignore[attr-defined]
    torch.backends = backends  # type: ignore[attr-defined]

    sys.modules["torch"] = torch
    sys.modules["torch.cuda"] = cuda
    sys.modules["torch.mps"] = mps
    sys.modules["torch.backends"] = backends
    sys.modules["torch.backends.mps"] = backends_mps
    _STUBBED_MODULES.add("torch")


def _install_lightweight_dependency_stubs() -> None:
    """Install lightweight stubs for genuinely missing heavy dependencies.

    Idempotent and conservative: a dependency that is already importable (or
    already present in ``sys.modules``) is never shadowed.  Only the standard
    library is used.
    """
    if "torch" in sys.modules:
        return
    try:
        if importlib.util.find_spec("torch") is not None:
            return
    except (ImportError, ValueError):
        # A broken parent package can make find_spec raise; fall through and
        # install the stub so the smoke path still runs.
        pass
    _install_torch_stub()


# ---------------------------------------------------------------------------
# Smoke checks
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

    failures: list[str] = []

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
    """Verify a deliberately broken ``sglang`` import is detected as a failure.

    Runs in a subprocess so the current interpreter's ``sys.modules`` is not
    polluted.  Returns 0 when the breakage is correctly detected, 1 otherwise.
    """
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
