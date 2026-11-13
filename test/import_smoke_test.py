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
import os
import sys

import pytest


def _ensure_source_on_path() -> None:
    """Make ``python/sglang`` importable from the source tree.

    The smoke tests must be runnable even when the ``sglang`` distribution is
    not installed (e.g. the CI environment-preparation step failed while
    resolving the heavy runtime dependency chain).  Adding the source
    directory to ``sys.path`` lets us exercise the real import chain without
    requiring a successful ``pip install``.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_dir = os.path.join(repo_root, "python")
    if os.path.isdir(os.path.join(python_dir, "sglang")):
        if python_dir not in sys.path:
            sys.path.insert(0, python_dir)


def _import_or_skip(module_name: str):
    """Import ``module_name``, skipping (with the original error) if the
    environment is not ready.

    A missing third-party dependency (e.g. torch not installed yet) is an
    environment problem, not a regression in the import chain under test.
    Skipping keeps the failure signal attributable: a real import error in
    ``sglang`` itself still raises and fails the test.

    ``ModuleNotFoundError`` is only treated as an environment problem when the
    missing module is *not* part of ``sglang`` itself; a missing ``sglang.*``
    submodule is a real regression and must fail the test.

    ``ImportError`` raised while importing a third-party dependency of
    ``sglang`` (e.g. a broken/partial install) is likewise reported as a skip
    so that a broken CI environment does not masquerade as a regression in the
    import chain under test.  Errors raised from ``sglang``'s own modules are
    re-raised so that a real regression still fails the test.
    """
    _ensure_source_on_path()
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing == "sglang" or missing.startswith("sglang."):
            raise
        pytest.skip(f"environment not ready: {exc!r}")
    except ImportError as exc:
        # Distinguish a failure originating inside sglang from one caused by a
        # missing/broken third-party dependency: only the latter is skipped.
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
