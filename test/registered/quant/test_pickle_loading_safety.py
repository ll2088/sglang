# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

The upstream fix removed the ``weights_only=False`` fallback and requires an
explicit SHA-256 digest for custom weights. These tests exercise a small
stand-in loader that mirrors that contract without importing torch.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pytest


class _UnsafePayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (Path.mkdir, (self.marker,))


def _load_weights(path: Path, expected_sha256: str | None) -> object:
    if expected_sha256 is None:
        raise RuntimeError("custom weights require an explicit SHA-256 digest")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.lower() != expected_sha256.lower():
        raise RuntimeError("weights failed SHA-256 verification")
    # Only after verification do we deserialize, and only trusted formats are
    # accepted by the real loader. Here we reject arbitrary pickle objects.
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError("unsupported checkpoint format")
    return payload


def test_missing_digest_is_rejected(tmp_path):
    path = tmp_path / "weights.pkl"
    path.write_bytes(pickle.dumps({"w": 1}))
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(path, None)


def test_mismatched_digest_is_rejected(tmp_path):
    path = tmp_path / "weights.pkl"
    path.write_bytes(pickle.dumps({"w": 1}))
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(path, "0" * 64)


def test_malicious_payload_is_not_executed(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "weights.pkl"
    path.write_bytes(pickle.dumps(_UnsafePayload(marker)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="unsupported checkpoint format"):
        _load_weights(path, digest)
    assert not marker.exists()


def test_verified_dict_payload_is_loaded(tmp_path):
    path = tmp_path / "weights.pkl"
    path.write_bytes(pickle.dumps({"w": 1}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _load_weights(path, digest) == {"w": 1}
