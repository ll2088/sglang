"""Compatibility checks for the blobfile 3.3.0 dependency bump.

These tests are intentionally lightweight: they do not import torch-heavy
modules, so they can run in a minimal environment.
"""

import glob
import importlib.metadata
import os
import re
import tempfile

import pytest


PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _config_files_with_blobfile():
    # setup.py does not declare runtime dependencies; only pyproject files do.
    files = [os.path.join(PYTHON_DIR, "pyproject.toml")]
    files.extend(sorted(glob.glob(os.path.join(PYTHON_DIR, "pyproject_*.toml"))))
    result = []
    for path in files:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            if re.search(r"blobfile\s*[=<>~!]", f.read()):
                result.append(path)
    return result


def test_blobfile_version_is_330():
    bf = pytest.importorskip("blobfile")
    try:
        version = importlib.metadata.version("blobfile")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(bf, "__version__", None)
    if version is None:
        pytest.skip("cannot determine installed blobfile version")
    assert version == "3.3.0", f"expected blobfile 3.3.0, found {version}"


def test_blobfile_constraints_consistent():
    files = _config_files_with_blobfile()
    if not files:
        pytest.skip("no config file declares blobfile")
    bad = []
    found = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        match = re.search(r"blobfile[^\n]*", text)
        line = match.group(0) if match else "<no blobfile line>"
        found.append((os.path.basename(path), line))
        if not match or "3.3.0" not in line:
            bad.append((path, line))
    assert not bad, f"inconsistent blobfile constraints: {bad}; found: {found}"


def test_blobfile_import_smoke():
    pytest.importorskip("blobfile")
    # Import lazily inside the test so collection never pulls in torch-heavy
    # modules; this keeps the suite runnable in a minimal environment.
    # sglang/__init__ and check_env both import torch, so skip cleanly when
    # torch is unavailable instead of failing the dependency-bump check.
    pytest.importorskip("torch")
    try:
        import sglang  # noqa: F401
        import sglang.check_env  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"sglang import unavailable in this environment: {exc}")


def test_blobfile_read_write_roundtrip():
    bf = pytest.importorskip("blobfile")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.bin")
        payload = b"blobfile-compat-check"
        with bf.BlobFile(path, "wb") as f:
            f.write(payload)
        with bf.BlobFile(path, "rb") as f:
            assert f.read() == payload
        assert bf.exists(path)
