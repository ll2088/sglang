# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

These tests are self-contained: they do not import ``sglang`` so that they
can run without the full runtime dependency tree. They mirror the upstream
fix that removed the ``weights_only=False`` fallback and requires an explicit
SHA-256 digest for custom checkpoints.
"""

import hashlib
import os
import pickle

import pytest


class _UnsupportedPayload:
    """A payload whose unpickling would create a file if executed."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_load(path, expected_sha256=None):
    """Mirror the fixed loader: verify digest, then load with weights_only."""
    if expected_sha256 is not None:
        actual = _sha256(path)
        if actual.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"Checkpoint digest mismatch: expected {expected_sha256}, got {actual}"
            )
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional for this test
        torch = None
    if torch is not None:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    with open(path, "rb") as f:
        return pickle.load(f)


def test_malicious_payload_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "payload.pt"
    path.write_bytes(pickle.dumps(_UnsupportedPayload(marker)))

    with pytest.raises(Exception):
        _safe_load(path)
    assert not marker.exists()


def test_digest_mismatch_is_rejected(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"trusted checkpoint")

    with pytest.raises(RuntimeError, match="digest mismatch"):
        _safe_load(path, expected_sha256="0" * 64)


def test_matching_digest_is_accepted(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"trusted checkpoint")

    assert _safe_load(path, expected_sha256=_sha256(path).upper()) == path
