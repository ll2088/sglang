"""Compatibility checks for the blobfile dependency (3.3.0).

These tests guard the dependency upgrade: they assert that blobfile is
importable, that its public API surface used by downstream code is present,
and that sglang.check_env can run without raising under the new version.
"""

import importlib.metadata

import pytest


def test_blobfile_importable():
    import blobfile  # noqa: F401


def test_blobfile_version_is_3_3_0():
    assert importlib.metadata.version("blobfile") == "3.3.0"


def test_blobfile_public_api_present():
    import blobfile

    # Core entry points that downstream code relies on.
    assert hasattr(blobfile, "BlobFile")
    assert hasattr(blobfile, "basename")
    assert hasattr(blobfile, "dirname")
    assert hasattr(blobfile, "join")
    assert hasattr(blobfile, "listdir")
    assert hasattr(blobfile, "copy")
    assert hasattr(blobfile, "exists")
    assert hasattr(blobfile, "glob")
    assert hasattr(blobfile, "local_path")


def test_blobfile_path_helpers():
    import blobfile

    assert blobfile.basename("gs://bucket/dir/file.txt") == "file.txt"
    assert blobfile.dirname("gs://bucket/dir/file.txt") == "gs://bucket/dir"
    assert blobfile.join("gs://bucket", "dir", "file.txt") == (
        "gs://bucket/dir/file.txt"
    )


def test_check_env_runs_without_error(capsys):
    from sglang import check_env

    check_env._get_env().check_env()
    out = capsys.readouterr().out
    assert "Python" in out
    assert "PyTorch" in out
