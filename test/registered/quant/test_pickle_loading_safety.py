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
    """Minimal safe loader: verify digest, then use ``pickle.loads`` only on
    bytes that match the expected digest. This mirrors the production contract
    that callers must supply a trusted digest for custom weights."""
    data = path.read_bytes()
    if expected_sha256 is None:
        raise RuntimeError("Refusing to load checkpoint without a SHA-256 digest")
    actual = hashlib.sha256(data).hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError("Checkpoint SHA-256 mismatch")
    return pickle.loads(data)


def test_malicious_payload_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "payload.pkl"
    path.write_bytes(pickle.dumps(_MaliciousPayload(marker)))
    with pytest.raises(RuntimeError, match="SHA-256"):
        _safe_load(path, expected_sha256="0" * 64)
    assert not marker.exists()


def test_missing_digest_is_rejected(tmp_path):
    path = tmp_path / "payload.pkl"
    path.write_bytes(pickle.dumps({"ok": True}))
    with pytest.raises(RuntimeError, match="SHA-256"):
        _safe_load(path)


def test_matching_digest_allows_load(tmp_path):
    path = tmp_path / "payload.pkl"
    path.write_bytes(pickle.dumps({"ok": True}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _safe_load(path, expected_sha256=digest) == {"ok": True}
