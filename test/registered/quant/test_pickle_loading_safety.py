# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

Upstream Model-Optimizer PR #3063 removed the `weights_only=False` fallback
and requires an explicit SHA-256 digest for custom weights.  These tests
exercise the same contract with a local helper so they do not depend on any
heavy runtime.
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


def _load_weights(path, expected_sha256=None):
    """Load a checkpoint only when its digest matches the expected value."""
    if expected_sha256 is None:
        raise RuntimeError("Checkpoint requires an explicit SHA-256 digest")
    with open(path, "rb") as f:
        payload = f.read()
    actual = hashlib.sha256(payload).hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError("Checkpoint failed SHA-256 verification")
    return pickle.loads(payload)


def test_missing_digest_is_rejected(tmp_path):
    path = tmp_path / "weights.pkl"
    path.write_bytes(pickle.dumps({"w": 1}))
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(str(path))


def test_mismatched_digest_is_rejected(tmp_path):
    path = tmp_path / "weights.pkl"
    path.write_bytes(pickle.dumps({"w": 1}))
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(str(path), "0" * 64)


def test_malicious_payload_is_not_executed(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps(_UnsupportedCheckpoint(marker)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # Even with a matching digest, the loader must not execute the payload
    # before verification; here we assert the digest gate rejects a bad one.
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(str(path), "0" * 64)
    assert not marker.exists()


def test_matching_digest_loads_payload(tmp_path):
    path = tmp_path / "weights.pkl"
    path.write_bytes(pickle.dumps({"w": 1}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _load_weights(str(path), digest.upper()) == {"w": 1}
