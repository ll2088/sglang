# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

These tests are self-contained (stdlib only) and mirror the upstream fix
that removed the ``weights_only=False`` fallback and requires an explicit
SHA-256 digest for custom weights.
"""

import hashlib
import pickle
import tempfile
from pathlib import Path

import pytest


class _MaliciousPayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (Path.touch, (self.marker,))


def _safe_load(path: Path, expected_sha256: str | None) -> bytes:
    """Minimal stand-in for the hardened loader.

    Rejects any payload unless a matching SHA-256 digest is supplied, and
    never falls back to ``pickle.load`` on untrusted input.
    """
    data = path.read_bytes()
    if expected_sha256 is None:
        raise RuntimeError("Refusing to load weights without a SHA-256 digest")
    actual = hashlib.sha256(data).hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError("Weights failed SHA-256 verification")
    return data


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    payload_path = tmp_path / "payload.pkl"
    payload_path.write_bytes(pickle.dumps(_MaliciousPayload(marker)))

    with pytest.raises(RuntimeError, match="SHA-256"):
        _safe_load(payload_path, expected_sha256=None)
    assert not marker.exists()


def test_wrong_digest_is_rejected(tmp_path):
    payload_path = tmp_path / "payload.pkl"
    payload_path.write_bytes(b"trusted checkpoint")
    with pytest.raises(RuntimeError, match="SHA-256"):
        _safe_load(payload_path, expected_sha256="0" * 64)


def test_matching_digest_is_accepted(tmp_path):
    payload_path = tmp_path / "payload.pkl"
    payload_path.write_bytes(b"trusted checkpoint")
    digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    assert _safe_load(payload_path, expected_sha256=digest.upper()) == b"trusted checkpoint"
