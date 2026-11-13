"""Lightweight smoke tests for the ``sglang`` top-level import chain.

These tests only depend on the standard library and pytest.  They are kept in
a standalone module (rather than inside ``test/run_suite.py``) so that pytest
can collect them directly without going through the CI suite runner's
argparse entry point or the registered-test sanity checks.

Each test imports the target module inside the test body so that an import
failure is reported as a test failure with the original traceback, instead of
aborting collection with an opaque error.
"""

import importlib
import importlib.metadata
import importlib.util
import os
import sys

import pytest

# NOTE: this module must stay importable with only the standard library and
# pytest.  It must never import torch/transformers/flashinfer/etc. at module
# scope; the heavy runtime dependency tree is stubbed by test/import_smoke.py
# so the smoke tests run even when ``pip install -e python`` failed during CI
# environment preparation.

# The smoke tests must stay runnable in a bare environment where the heavy
# sglang runtime dependency tree was never installed.  ``pytest`` itself is
# the only third-party import at module scope; everything else is stdlib.

# Reuse the self-contained bootstrap helpers from test/import_smoke.py so the
# two smoke entry points cannot drift apart.  Imported by file path so this
# module does not depend on ``test`` being an importable package (it has no
# __init__.py) nor on the repository root being on sys.path.
_smoke_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "import_smoke.py")
_smoke_spec = importlib.util.spec_from_file_location("import_smoke", _smoke_path)
assert _smoke_spec is not None and _smoke_spec.loader is not None
_smoke = importlib.util.module_from_spec(_smoke_spec)
sys.modules.setdefault("import_smoke", _smoke)
_smoke_spec.loader.exec_module(_smoke)

_ensure_source_on_path = _smoke._ensure_source_on_path
_install_lightweight_dependency_stubs = _smoke._install_lightweight_dependency_stubs
_stub_is_installed = _smoke._stub_is_installed
_run_bare_smoke = _smoke._run_bare_smoke
_run_selfcheck = _smoke._run_selfcheck
_run_negative_check = _smoke._run_negative_check

# Bootstrap the source tree and lightweight dependency stubs as early as
# possible -- before any test module (including test/run_suite.py) imports
# ``sglang``.  This keeps the smoke tests runnable in a bare environment where
# the full sglang dependency tree was never installed, so a failed
# environment-preparation step cannot mask a real import regression.
_ensure_source_on_path()
_install_lightweight_dependency_stubs()


def _import_or_skip(module_name: str):
    """Import ``module_name``, skipping (with the original error) if the
    environment is not ready.

    The smoke tests must not require the ``sglang`` distribution to be
    installed: ``python/`` is injected onto ``sys.path`` by
    ``_ensure_source_on_path`` so the source tree is imported directly.

    A missing third-party dependency (e.g. torch not installed yet) is an
    environment problem, not a regression in the import chain under test.
    Skipping keeps the failure signal attributable: a real import error in
    ``sglang`` itself still raises and fails the test.

    ``ModuleNotFoundError`` is only treated as an environment problem when the
    missing module is *not* part of ``sglang`` itself; a missing ``sglang.*``
    submodule is a real regression and must fail the test.
    """
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
def test_import_smoke_bare_entrypoint_isolated():
    """Smoke test: the bare entry point must pass in a fresh subprocess.

    Guards against the smoke test passing only because an earlier test in
    the same interpreter already imported ``sglang`` (or installed stubs).
    """
    assert _run_bare_smoke() == 0


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
    """Smoke test: sglang.__version__ must match installed package metadata.

    The smoke tests are intentionally runnable without the ``sglang``
    distribution installed (see ``_ensure_source_on_path``), so this check
    skips when the distribution metadata is absent instead of failing.
    """
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


@pytest.mark.import_smoke
def test_lightweight_dependency_stubs_effective():
    """Self-check: the bare-environment bootstrap must actually take effect."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()

    if importlib.util.find_spec("torch") is not None and not _stub_is_installed(
        "torch"
    ):
        pytest.skip("real torch is installed; stub intentionally not used")

    assert _stub_is_installed("torch"), (
        "lightweight torch stub was not installed; the bare-environment "
        "bootstrap is broken"
    )


@pytest.mark.import_smoke
def test_import_smoke_without_installed_distribution():
    """Smoke test: the import chain must work without ``pip install sglang``."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_dir = os.path.join(repo_root, "python")
    assert python_dir in sys.path, (
        "python/ must be on sys.path so the smoke tests do not require an "
        "installed sglang distribution"
    )

    sglang = _import_or_skip("sglang")
    assert sglang.__version__


@pytest.mark.import_smoke
def test_import_smoke_bare_entrypoint():
    """Smoke test: the ``--bare`` entry point must succeed without pytest."""
    assert _run_bare_smoke() == 0


@pytest.mark.import_smoke
def test_import_smoke_selfcheck():
    """Smoke test: the bootstrap self-check must pass."""
    assert _run_selfcheck() == 0


@pytest.mark.import_smoke
def test_import_smoke_negative():
    """Negative smoke test: a broken top-level import must be detected."""
    assert _run_negative_check() == 0


if __name__ == "__main__":
    if "--bare" in sys.argv[1:]:
        raise SystemExit(_run_bare_smoke())
    if "--selfcheck" in sys.argv[1:]:
        raise SystemExit(_run_selfcheck())
    if "--negative" in sys.argv[1:]:
        raise SystemExit(_run_negative_check())
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    raise SystemExit(
        pytest.main(["-q", os.path.abspath(__file__), "-k", "import_smoke"])
    )
