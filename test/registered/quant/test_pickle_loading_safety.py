# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe checkpoint loading.

Upstream removed a `weights_only=False` fallback and now verifies cached
weights with SHA-256. These tests exercise the same contract with a small
self-contained loader so they run without torch or sglang installed.
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_verified(path: Path, expected_sha256: str):
    """Load a checkpoint only after verifying its digest.

    Mirrors the upstream contract: a digest is mandatory, and a mismatch
    raises before any deserialization happens.
    """
    if not expected_sha256:
        raise RuntimeError("A SHA-256 digest is required to load weights")
    actual = _sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Checkpoint digest mismatch: expected {expected_sha256}, got {actual}"
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def test_missing_digest_is_rejected(tmp_path):
    path = tmp_path / "weights.pkl"
    path.write_bytes(pickle.dumps({"w": 1}))
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_verified(path, "")


def test_digest_mismatch_is_rejected_before_deserialization(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps(_UnsafePayload(marker)))
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _load_verified(path, "0" * 64)
    assert not marker.exists()


def test_matching_digest_loads_payload(tmp_path):
    path = tmp_path / "weights.pkl"
    path.write_bytes(pickle.dumps({"w": 1}))
    digest = _sha256(path)
    assert _load_verified(path, digest.upper()) == {"w": 1}


def test_tampered_cached_weights_are_rejected(tmp_path):
    path = tmp_path / "cached.pkl"
    path.write_bytes(pickle.dumps({"w": 1}))
    digest = _sha256(path)
    path.write_bytes(pickle.dumps({"w": 2}))
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _load_verified(path, digest)
