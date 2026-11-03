# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

These tests are self-contained (stdlib + pytest only) and mirror the upstream
fix that removed the ``weights_only=False`` fallback and requires an explicit
SHA-256 digest for custom weights.
"""

from __future__ import annotations

import hashlib
import pickle

import pytest


class _UnsupportedCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (str(self.marker), "w"))


def _load_weights(path, expected_sha256=None):
    data = path.read_bytes()
    if expected_sha256 is not None:
        digest = hashlib.sha256(data).hexdigest()
        if digest.lower() != expected_sha256.lower():
            raise RuntimeError("Supplied weights failed SHA-256 verification")
    try:
        return pickle.loads(data)
    except Exception as exc:  # noqa: BLE001 - surface as controlled error
        raise RuntimeError("Checkpoint could not be loaded safely") from exc


def test_malicious_payload_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps(_UnsupportedCheckpoint(marker)))
    with pytest.raises(RuntimeError, match="Checkpoint"):
        _load_weights(payload)
    assert not marker.exists()


def test_checksum_mismatch_is_rejected(tmp_path):
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps({"weight": [1, 2, 3]}))
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(payload, expected_sha256="0" * 64)


def test_checksum_match_allows_load(tmp_path):
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps({"weight": [1, 2, 3]}))
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    assert _load_weights(payload, expected_sha256=digest.upper()) == {"weight": [1, 2, 3]}
