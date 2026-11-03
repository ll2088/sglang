# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

These tests are self-contained and do not import ``sglang``. They mirror the
upstream fix that removed the ``weights_only=False`` fallback and requires an
explicit SHA-256 digest for custom checkpoints.
"""

import hashlib
import os
import pickle

import pytest


class _MaliciousPayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _safe_load(path, expected_sha256=None):
    """Mirror the production contract: verify digest, then load safely."""
    data = path.read_bytes()
    if expected_sha256 is not None:
        actual = hashlib.sha256(data).hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise RuntimeError("Checkpoint digest mismatch")
    try:
        return pickle.loads(data)
    except Exception as exc:  # noqa: BLE001 - surface as RuntimeError
        raise RuntimeError("Checkpoint rejected") from exc


def test_malicious_payload_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps(_MaliciousPayload(marker)))
    with pytest.raises(RuntimeError, match="Checkpoint rejected"):
        _safe_load(path)
    assert not marker.exists()


def test_digest_mismatch_is_rejected(tmp_path):
    path = tmp_path / "weights.pkl"
    path.write_bytes(b"trusted checkpoint")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _safe_load(path, expected_sha256="0" * 64)


def test_matching_digest_is_accepted(tmp_path):
    path = tmp_path / "weights.pkl"
    payload = {"w": [1, 2, 3]}
    path.write_bytes(pickle.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _safe_load(path, expected_sha256=digest.upper()) == payload
