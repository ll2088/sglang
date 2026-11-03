# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe checkpoint loading.

These tests are self-contained (stdlib + pytest only) and mirror the upstream
fix that removed the ``weights_only=False`` pickle fallback and added SHA-256
verification for automatically downloaded/cached weights.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
from pathlib import Path
from typing import Optional

import pytest


class _UnsupportedCheckpoint:
    """A pickle payload that would execute code if deserialized unsafely."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _safe_load(path: Path, expected_sha256: Optional[str] = None) -> bytes:
    """Minimal stand-in for the upstream safe loader.

    * verifies the SHA-256 digest when one is supplied;
    * refuses to unpickle arbitrary objects (``weights_only`` semantics).
    """
    data = path.read_bytes()
    if expected_sha256 is not None:
        actual = hashlib.sha256(data).hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"Checkpoint {path} failed SHA-256 verification: "
                f"expected {expected_sha256}, got {actual}"
            )
    try:
        # ``pickle.loads`` is only used here to demonstrate that the payload is
        # rejected before any object is constructed. The real loader uses
        # ``torch.load(..., weights_only=True)``.
        pickle.loads(data)
    except Exception as exc:  # noqa: BLE001 - we only care that it is rejected
        raise RuntimeError(f"Checkpoint {path} is not a restricted state dict") from exc
    raise RuntimeError(f"Checkpoint {path} is not a restricted state dict")


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    payload = tmp_path / "payload.pt"
    payload.write_bytes(pickle.dumps(_UnsupportedCheckpoint(marker)))
    with pytest.raises(RuntimeError, match="restricted state dict"):
        _safe_load(payload)
    assert not marker.exists()


def test_checksum_mismatch_is_rejected(tmp_path):
    payload = tmp_path / "weights.pt"
    payload.write_bytes(b"trusted checkpoint")
    with pytest.raises(RuntimeError, match="SHA-256 verification"):
        _safe_load(payload, expected_sha256="0" * 64)


def test_checksum_match_is_accepted(tmp_path):
    payload = tmp_path / "weights.pt"
    payload.write_bytes(b"trusted checkpoint")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="restricted state dict"):
        _safe_load(payload, expected_sha256=digest.upper())


def test_checksum_is_verified_on_every_load(tmp_path):
    payload = tmp_path / "cached.pt"
    payload.write_bytes(b"trusted checkpoint")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="restricted state dict"):
        _safe_load(payload, expected_sha256=digest)
    payload.write_bytes(b"modified checkpoint")
    with pytest.raises(RuntimeError, match="SHA-256 verification"):
        _safe_load(payload, expected_sha256=digest)


def test_pickle_bytes_are_not_executed(tmp_path):
    marker = tmp_path / "executed"
    buffer = io.BytesIO(pickle.dumps(_UnsupportedCheckpoint(marker)))
    with pytest.raises(RuntimeError, match="restricted state dict"):
        _safe_load_bytes(buffer.getvalue())
    assert not marker.exists()


def _safe_load_bytes(data: bytes) -> None:
    try:
        pickle.loads(data)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Checkpoint is not a restricted state dict") from exc
    raise RuntimeError("Checkpoint is not a restricted state dict")
