# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe checkpoint loading.

Verifies that a malicious pickle payload is rejected without executing code,
and that SHA-256 verification rejects tampered weights.
"""

from __future__ import annotations

import hashlib
import os

import pytest


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(path, expected_sha256: str) -> None:
    actual = _sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Checkpoint {path} failed SHA-256 verification: "
            f"expected {expected_sha256}, got {actual}"
        )


def _safe_load(path, expected_sha256: str):
    _verify(path, expected_sha256)
    # Only allow a restricted, non-executable format.
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


class _MaliciousPayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    torch = pytest.importorskip("torch")
    marker = tmp_path / "executed"
    path = tmp_path / "evil.pt"
    torch.save(_MaliciousPayload(marker), path)
    digest = _sha256(path)
    with pytest.raises(Exception):
        _safe_load(path, digest)
    assert not marker.exists()


def test_tampered_weights_fail_checksum(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"trusted checkpoint")
    digest = _sha256(path)
    _verify(path, digest)
    path.write_bytes(b"modified checkpoint")
    with pytest.raises(RuntimeError, match="SHA-256"):
        _verify(path, digest)


def test_checksum_comparison_is_case_insensitive(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"payload")
    _verify(path, _sha256(path).upper())
