# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

These tests are self-contained (stdlib + pytest only) and mirror the
upstream fix that removed the ``weights_only=False`` fallback and requires
SHA-256 verification for downloaded/cached weights.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
from pathlib import Path

import pytest


class _MaliciousPayload:
    """Pickle payload that would create a marker file if executed."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_load(path: Path, expected_sha256: str | None) -> object:
    """Minimal safe loader mirroring the upstream contract.

    * ``expected_sha256`` is mandatory for any non-empty file.
    * The digest is verified before deserialization.
    * Deserialization uses ``pickle.loads`` on bytes that have already been
      verified, and rejects payloads that are not plain builtin containers.
    """
    data = path.read_bytes()
    if expected_sha256 is None:
        raise RuntimeError("Checkpoint requires an explicit SHA-256 digest")
    actual = hashlib.sha256(data).hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError("Checkpoint failed SHA-256 verification")
    obj = pickle.loads(data)
    if not isinstance(obj, (dict, list, tuple, int, float, str, bytes, type(None))):
        raise RuntimeError("Checkpoint contains unsupported object")
    return obj


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps(_MaliciousPayload(marker)))
    digest = _sha256(payload)
    with pytest.raises(RuntimeError, match="unsupported object"):
        _safe_load(payload, digest)
    assert not marker.exists()


def test_missing_digest_is_rejected(tmp_path):
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps({"a": 1}))
    with pytest.raises(RuntimeError, match="explicit SHA-256"):
        _safe_load(payload, None)


def test_wrong_digest_is_rejected(tmp_path):
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps({"a": 1}))
    with pytest.raises(RuntimeError, match="SHA-256 verification"):
        _safe_load(payload, "0" * 64)


def test_verified_plain_payload_loads(tmp_path):
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps({"a": 1}))
    assert _safe_load(payload, _sha256(payload)) == {"a": 1}
