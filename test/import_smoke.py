"""Self-contained bootstrap helpers for the sglang import smoke tests.

This module only depends on the standard library.  It is imported by file
path from ``test/import_smoke_test.py`` and ``test/run_suite.py`` so the two
smoke entry points share one implementation and cannot drift apart.

The helpers here make the smoke tests runnable in a bare environment where
the full ``sglang`` dependency tree (torch, py-spy, pybase64, pydantic,
pyzmq, quack-kernels, regex, ...) was never installed -- so a failed
environment-preparation step cannot mask a real import regression.

Run directly with::

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

    Idempotent: repeated calls do not duplicate the entry.  Never shadows a
    real installation -- the source tree is prepended only when it is not
    already present.
    """
    python_dir = os.path.join(_repo_root(), "python")
    if os.path.isdir(python_dir) and python_dir not in sys.path:
        sys.path.insert(0, python_dir)


# ---------------------------------------------------------------------------
# Lightweight dependency stubs
# ---------------------------------------------------------------------------

_STUB_MARKER = "__sglang_import_smoke_stub__"


def _stub_is_installed(module_name: str) -> bool:
    """Return True when ``module_name`` is one of our lightweight stubs."""
    mod = sys.modules.get(module_name)
    return mod is not None and getattr(mod, _STUB_MARKER, False) is True


def _make_stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    setattr(mod, _STUB_MARKER, True)
    mod.__path__ = []  # type: ignore[attr-defined]
    mod.__package__ = name
    return mod


def _install_lightweight_dependency_stubs() -> None:
    """Install minimal stubs for genuinely missing heavy dependencies.

    Only stubs dependencies that are *not* importable in the current
    environment, so a real installation is never shadowed.  Idempotent.
    """
    _ensure_source_on_path()

    # torch: the top-level sglang import chain pulls it in via
    # hf_transformers_patches / lang.api.  A stub is enough for the smoke
    # checks (they only assert the import chain completes and __version__ is
    # readable).
    if importlib.util.find_spec("torch") is None and not _stub_is_installed(
        "torch"
    ):
        torch = _make_stub_module("torch")

        class _Device:
            def __init__(self, spec=None, type=None):  # noqa: A002
                self.type = type or "cpu"
                self.index = None

            def __str__(self):
                return self.type

        def _device(spec=None, type=None):  # noqa: A002
            if isinstance(spec, _Device):
                return spec
            return _Device(spec, type)

        torch.device = _device  # type: ignore[attr-defined]
        torch.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
        torch.dtype = type("dtype", (), {})  # type: ignore[attr-defined]
        torch.float16 = torch.dtype()  # type: ignore[attr-defined]
        torch.bfloat16 = torch.dtype()  # type: ignore[attr-defined]
        torch.float32 = torch.dtype()  # type: ignore[attr-defined]

        cuda = _make_stub_module("torch.cuda")
        cuda.is_available = lambda: False  # type: ignore[attr-defined]
        cuda.device_count = lambda: 0  # type: ignore[attr-defined]
        cuda.current_device = lambda: 0  # type: ignore[attr-defined]
        cuda.get_device_properties = lambda *a, **k: None  # type: ignore[attr-defined]
        torch.cuda = cuda  # type: ignore[attr-defined]

        mps = _make_stub_module("torch.mps")
        mps.is_available = lambda: False  # type: ignore[attr-defined]
        mps.recommended_max_memory = lambda: 0  # type: ignore[attr-defined]
        mps.current_allocated_memory = lambda: 0  # type: ignore[attr-defined]
        mps.driver_allocated_memory = lambda: 0  # type: ignore[attr-defined]
        torch.mps = mps  # type: ignore[attr-defined]

        backends = _make_stub_module("torch.backends")
        backends.mps = mps  # type: ignore[attr-defined]
        torch.backends = backends  # type: ignore[attr-defined]

        sys.modules["torch"] = torch
        sys.modules["torch.cuda"] = cuda
        sys.modules["torch.mps"] = mps
        sys.modules["torch.backends"] = backends

    # triton: only imported on Apple Silicon via _platform_stubs; stub it so
    # the import chain does not fail on a bare Linux CI box either.
    if importlib.util.find_spec("triton") is None and not _stub_is_installed(
        "triton"
    ):
        triton = _make_stub_module("triton")
        triton.__version__ = "0.0.0-stub"  # type: ignore[attr-defined]
        sys.modules["triton"] = triton


# ---------------------------------------------------------------------------
# Bare / selfcheck / negative entry points
# ---------------------------------------------------------------------------


def _run_bare_smoke() -> int:
    """Run the minimal smoke check without pytest.

    Returns 0 on success, 1 on failure.  Prints one greppable line so the CI
    log carries an explicit signal.
    """
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
    """Run every smoke check without pytest and without third-party deps.

    Returns 0 when every check passes, 1 otherwise.
    """
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

    for module_name in ("sglang._platform_stubs", "sglang.version", "sglang.utils"):
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
    """Negative self-check: a broken ``sglang`` import must be reported.

    Runs in a subprocess so the current interpreter's ``sys.modules`` is not
    polluted by the deliberately broken import.  Returns 0 when the broken
    import is correctly detected as a failure, 1 otherwise.
    """
    import subprocess

    repo_root = _repo_root()
    python_dir = os.path.join(repo_root, "python")
    init_path = os.path.join(python_dir, "sglang", "__init__.py")
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
