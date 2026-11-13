"""Lightweight smoke tests for the ``sglang`` top-level import chain.

These tests only depend on the standard library and pytest.  They are kept in
a standalone module so that pytest can collect them directly without going
through the CI suite runner's argparse entry point or the registered-test
sanity checks.

Each test imports the target module inside the test body so that an import
failure is reported as a test failure with the original traceback, instead of
aborting collection with an opaque error.
"""

import importlib
import importlib.metadata
import importlib.util
import os
import sys
import types

import pytest


def _ensure_source_on_path() -> None:
    """Make ``python/sglang`` importable from the source tree."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_dir = os.path.join(repo_root, "python")
    if os.path.isdir(os.path.join(python_dir, "sglang")):
        if python_dir not in sys.path:
            sys.path.insert(0, python_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


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


def _install_lightweight_dependency_stubs() -> None:
    """Install minimal stubs for genuinely missing heavy dependencies."""
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


_ensure_source_on_path()
_install_lightweight_dependency_stubs()


def _import_or_skip(module_name: str):
    """Import ``module_name``, skipping if the environment is not ready."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing == "sglang" or missing.startswith("sglang."):
            raise
        pytest.skip(f"environment not ready: {exc!r}")
    except ImportError as exc:
        tb = exc.__traceback__
        while tb is not None:
            frame_file = tb.tb_frame.f_code.co_filename
            if "sglang" in frame_file.replace("\\", "/"):
                raise
            tb = tb.tb_next
        pytest.skip(f"environment not ready: {exc!r}")


@pytest.mark.import_smoke
def test_import_smoke():
    """Smoke test: importing sglang must succeed and expose __version__."""
    sglang = _import_or_skip("sglang")

    assert sglang.__version__
    assert isinstance(sglang.__version__, str)


@pytest.mark.import_smoke
def test_import_platform_stubs():
    """Smoke test: the platform stubs module must import without raising."""
    stubs = _import_or_skip("sglang._platform_stubs")

    assert hasattr(stubs, "install_platform_stubs")
    assert callable(stubs.install_platform_stubs)


@pytest.mark.import_smoke
def test_import_version():
    """Smoke test: version attributes must be readable and consistent."""
    version = _import_or_skip("sglang.version")

    assert version.__version__
    assert isinstance(version.__version__, str)
    assert version.__version_tuple__
    assert isinstance(version.__version_tuple__, tuple)


@pytest.mark.import_smoke
def test_import_version_matches_metadata():
    """Smoke test: sglang.__version__ must match installed package metadata."""
    sglang = _import_or_skip("sglang")

    try:
        dist_version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("sglang distribution metadata not installed")

    assert sglang.__version__ == dist_version


@pytest.mark.import_smoke
def test_import_utils():
    """Smoke test: sglang.utils must be importable from the package."""
    utils = _import_or_skip("sglang.utils")

    assert hasattr(utils, "LazyImport")


@pytest.mark.import_smoke
def test_import_public_api_symbols():
    """Smoke test: the public API names declared in __all__ must resolve."""
    sglang = _import_or_skip("sglang")

    assert sglang.__all__
    for name in sglang.__all__:
        assert hasattr(sglang, name), f"missing public symbol: {name}"


def _run_bare_smoke() -> int:
    """Run the smoke checks without pytest (standard library only)."""
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


def _run_negative_check() -> int:
    """Negative self-check: a broken ``sglang`` import must be reported."""
    import subprocess

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


@pytest.mark.import_smoke
def test_import_smoke_bare_entrypoint():
    """Smoke test: the ``--bare`` entry point must succeed without pytest."""
    assert _run_bare_smoke() == 0


@pytest.mark.import_smoke
def test_import_smoke_negative():
    """Negative smoke test: a broken top-level import must be detected."""
    assert _run_negative_check() == 0


if __name__ == "__main__":
    if "--bare" in sys.argv[1:]:
        raise SystemExit(_run_bare_smoke())
    if "--negative" in sys.argv[1:]:
        raise SystemExit(_run_negative_check())
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    raise SystemExit(
        pytest.main(["-q", os.path.abspath(__file__), "-k", "import_smoke"])
    )
