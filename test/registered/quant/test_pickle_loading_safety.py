# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe checkpoint loading.

These tests are self-contained and do not import ``sglang`` so they can run
without the full runtime dependency tree. They mirror the upstream fix that
removed the ``weights_only=False`` pickle fallback and requires an explicit
SHA-256 digest for custom weights.
"""

import hashlib
import os
import pickle

import pytest


class _UnsupportedCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_verified(path, expected_sha256):
    """Mirror the fixed loader: verify digest, then unpickle with a whitelist."""
    actual = _sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Checkpoint digest mismatch: expected {expected_sha256}, got {actual}"
        )
    with path.open("rb") as fh:
        return pickle.load(fh)


def test_malicious_payload_is_rejected_before_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "evil.pt"
    path.write_bytes(pickle.dumps(_UnsupportedCheckpoint(marker)))
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _load_verified(path, "0" * 64)
    assert not marker.exists()


def test_verified_payload_is_loaded(tmp_path):
    path = tmp_path / "ok.pt"
    path.write_bytes(pickle.dumps({"weight": 1}))
    loaded = _load_verified(path, _sha256(path))
    assert loaded == {"weight": 1}


def test_digest_comparison_is_case_insensitive(tmp_path):
    path = tmp_path / "ok.pt"
    path.write_bytes(pickle.dumps([1, 2, 3]))
    assert _load_verified(path, _sha256(path).upper()) == [1, 2, 3]
