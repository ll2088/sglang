"""Compatibility checks for the blobfile dependency (3.3.0).

blobfile is a transitive/declared dependency of sglang; this module guards
against import-time or basic-API regressions when the pin is bumped.
"""

import importlib
import importlib.metadata
import re
from pathlib import Path

import pytest


def _declared_blobfile_pin():
    """Return the blobfile pin declared in the nearest pyproject.toml, if any."""
    for candidate in Path(__file__).resolve().parents:
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            match = re.search(
                r'"blobfile==([^"]+)"', pyproject.read_text(encoding="utf-8")
            )
            if match is not None:
                return match.group(1)
    return None


def test_blobfile_importable():
    module = importlib.import_module("blobfile")
    assert module is not None


def test_blobfile_version_metadata_consistent():
    version = importlib.metadata.version("blobfile")
    assert version, "blobfile distribution metadata must expose a version"
    # The declared pin in the project metadata must match the installed one.
    # Compare against the pin declared in pyproject.toml instead of hardcoding
    # a major version, so this check stays valid across future bumps.
    declared = _declared_blobfile_pin()
    if declared is not None:
        assert version.split("+")[0] == declared, (version, declared)


def test_blobfile_blobfile_class_available():
    blobfile = importlib.import_module("blobfile")
    assert hasattr(blobfile, "BlobFile")
    assert callable(blobfile.BlobFile)


def test_blobfile_local_file_roundtrip(tmp_path):
    blobfile = importlib.import_module("blobfile")
    target = tmp_path / "payload.bin"
    payload = b"blobfile-compat-check"
    with blobfile.BlobFile(str(target), "wb") as handle:
        handle.write(payload)
    with blobfile.BlobFile(str(target), "rb") as handle:
        assert handle.read() == payload


def test_blobfile_public_helpers_present():
    blobfile = importlib.import_module("blobfile")
    for name in ("BlobFile", "LocalBlobFile", "basename", "dirname"):
        assert hasattr(blobfile, name), f"blobfile.{name} is missing"


def test_sglang_import_and_version_metadata():
    sglang = pytest.importorskip("sglang")
    assert getattr(sglang, "__version__", None)
    try:
        installed = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("sglang distribution metadata is not installed")
    assert installed
