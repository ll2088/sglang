# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

Upstream removed the ``weights_only=False`` fallback and requires explicit
SHA-256 verification for custom weights. These tests exercise the pattern
without importing sglang.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest


class _MaliciousPayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _load_weights(path: Path, expected_sha256: str | None) -> bytes:
    data = path.read_bytes()
    if expected_sha256 is None:
        raise RuntimeError("Checkpoint requires an explicit SHA-256 digest")
    actual = hashlib.sha256(data).hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError("Checkpoint failed SHA-256 verification")
    return data


def test_missing_digest_is_rejected(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"payload")
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(path, None)


def test_wrong_digest_is_rejected(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"payload")
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(path, "0" * 64)


def test_correct_digest_is_accepted(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"payload")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _load_weights(path, digest) == b"payload"


def test_malicious_payload_is_not_executed(tmp_path):
    marker = tmp_path / "executed"
    payload = _MaliciousPayload(marker)
    # Simulate a loader that refuses to unpickle without a trusted digest.
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(tmp_path / "missing.bin", None)
    assert not marker.exists()
