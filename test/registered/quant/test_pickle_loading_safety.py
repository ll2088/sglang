# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

These tests are self-contained and do not import ``sglang`` so that they can
run with only pytest installed. They mirror the upstream fix that removes the
``weights_only=False`` fallback and requires an explicit SHA-256 digest for
custom checkpoints.
"""

import hashlib
import os
import pickle

import pytest


class _MaliciousPayload:
    """Pickle payload that would create a marker file if executed."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_load(path, expected_sha256=None):
    """Mirror the fixed loading contract: verify digest, then load safely."""
    if expected_sha256 is None:
        raise RuntimeError("Custom checkpoint requires an explicit SHA-256 digest")
    actual = _sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError("Checkpoint failed SHA-256 verification")
    with open(path, "rb") as f:
        return pickle.load(f)


def test_malicious_payload_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "malicious.pkl"
    path.write_bytes(pickle.dumps(_MaliciousPayload(marker)))
    with pytest.raises(RuntimeError, match="SHA-256"):
        _safe_load(path, expected_sha256="0" * 64)
    assert not marker.exists()


def test_missing_digest_is_rejected(tmp_path):
    path = tmp_path / "weights.pkl"
    path.write_bytes(pickle.dumps({"w": 1}))
    with pytest.raises(RuntimeError, match="explicit SHA-256"):
        _safe_load(path)


def test_matching_digest_allows_load(tmp_path):
    path = tmp_path / "weights.pkl"
    path.write_bytes(pickle.dumps({"w": 1}))
    assert _safe_load(path, expected_sha256=_sha256(path)) == {"w": 1}


def test_digest_comparison_is_case_insensitive(tmp_path):
    path = tmp_path / "weights.pkl"
    path.write_bytes(pickle.dumps({"w": 2}))
    assert _safe_load(path, expected_sha256=_sha256(path).upper()) == {"w": 2}
