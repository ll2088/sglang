"""Lightweight smoke tests for the ``sglang`` top-level import chain.

These tests only depend on the standard library and pytest.  They are kept in
a standalone module (rather than inside ``test/run_suite.py``) so that pytest
can collect them directly without going through the CI suite runner's
argparse entry point or the registered-test sanity checks.
"""

import importlib.metadata

import pytest


@pytest.mark.import_smoke
def test_import_smoke():
    """Smoke test: importing sglang must succeed and expose __version__."""
    import sglang

    assert sglang.__version__
    assert isinstance(sglang.__version__, str)


@pytest.mark.import_smoke
def test_import_platform_stubs():
    """Smoke test: the platform stubs module must import without raising."""
    import sglang._platform_stubs as stubs

    assert hasattr(stubs, "install_platform_stubs")
    assert callable(stubs.install_platform_stubs)


@pytest.mark.import_smoke
def test_import_version():
    """Smoke test: version attributes must be readable and consistent."""
    from sglang import version

    assert version.__version__
    assert isinstance(version.__version__, str)
    assert version.__version_tuple__
    assert isinstance(version.__version_tuple__, tuple)


@pytest.mark.import_smoke
def test_import_version_matches_metadata():
    """Smoke test: sglang.__version__ must match installed package metadata."""
    import sglang

    try:
        dist_version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("sglang distribution metadata not installed")

    assert sglang.__version__ == dist_version


@pytest.mark.import_smoke
def test_import_utils():
    """Smoke test: sglang.utils must be importable from the package."""
    from sglang import utils

    assert hasattr(utils, "LazyImport")
