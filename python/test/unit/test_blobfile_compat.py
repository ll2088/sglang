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


BLOBFILE_RE = re.compile(r"blobfile\s*([=<>~!][^\s\"',]*)")


def _all_config_files():
    # setup.py does not declare runtime dependencies, but it is included so the
    # consistency check can assert it does not silently declare blobfile.
    files = [os.path.join(PYTHON_DIR, "setup.py"), os.path.join(PYTHON_DIR, "pyproject.toml")]
    files.extend(sorted(glob.glob(os.path.join(PYTHON_DIR, "pyproject_*.toml"))))
    return [path for path in files if os.path.isfile(path)]


def _blobfile_constraint(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        match = BLOBFILE_RE.search(f.read())
    return match.group(1) if match else None


def _config_files_with_blobfile():
    result = []
    for path in _all_config_files():
        if _blobfile_constraint(path) is not None:
            result.append(path)
    return result


def test_blobfile_version_is_330():
    try:
        bf = importlib.import_module("blobfile")
    except ImportError as exc:
        pytest.fail(f"blobfile is not importable: {exc}")
    try:
        version = importlib.metadata.version("blobfile")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(bf, "__version__", None)
    assert version is not None, "cannot determine installed blobfile version"
    assert version == "3.3.0", f"expected blobfile 3.3.0, found {version}"


def test_blobfile_constraints_consistent():
    files = _config_files_with_blobfile()
    assert files, "no config file declares blobfile"
    constraints = {os.path.basename(path): _blobfile_constraint(path) for path in files}
    distinct = set(constraints.values())
    assert distinct == {"==3.3.0"}, (
        f"blobfile constraints must all be '==3.3.0', found: {constraints}"
    )
    # setup.py must not declare a runtime blobfile dependency (it has none).
    setup_py = os.path.join(PYTHON_DIR, "setup.py")
    assert _blobfile_constraint(setup_py) is None, (
        "python/setup.py unexpectedly declares a blobfile constraint"
    )


def test_blobfile_import_smoke():
    try:
        bf = importlib.import_module("blobfile")
    except ImportError as exc:
        pytest.fail(f"blobfile is not importable: {exc}")
    # Keep this check independent of torch: sglang/__init__ and check_env pull
    # in torch and FlashInfer, which are not installable in the minimal CI
    # environment used for the dependency-bump check. Importing blobfile and
    # touching its public API is what actually guards against API removals or
    # renames in the 3.0.0 -> 3.3.0 bump.
    for name in ("BlobFile", "LocalBlobFile", "BlobFileWriter"):
        assert hasattr(bf, name), f"blobfile is missing public API {name}"
    assert callable(bf.exists)


def test_blobfile_read_write_roundtrip():
    try:
        bf = importlib.import_module("blobfile")
    except ImportError as exc:
        pytest.fail(f"blobfile is not importable: {exc}")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.bin")
        payload = b"blobfile-compat-check"
        with bf.BlobFile(path, "wb") as f:
            f.write(payload)
        with bf.BlobFile(path, "rb") as f:
            assert f.read() == payload
        assert bf.exists(path)
