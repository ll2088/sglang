# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe checkpoint loading.

A malicious pickle payload must be rejected without executing its
``__reduce__`` callable. The helper below mirrors the upstream pattern:
``torch.load(..., weights_only=True)`` first, and any fallback must be
explicitly gated by a caller-supplied checksum.
"""

import hashlib
import os

import pytest


torch = pytest.importorskip("torch")


class _UnsupportedCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _load_weights(path, expected_sha256=None):
    if expected_sha256 is not None:
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if digest.lower() != expected_sha256.lower():
            raise RuntimeError("Checkpoint checksum mismatch")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError("Checkpoint rejected by safe loader") from exc


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "evil.pt"
    torch.save(_UnsupportedCheckpoint(marker), path)
    with pytest.raises(RuntimeError, match="Checkpoint"):
        _load_weights(str(path))
    assert not marker.exists()


def test_checksum_mismatch_is_rejected(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"w": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError, match="checksum"):
        _load_weights(str(path), expected_sha256="0" * 64)


def test_checksum_match_allows_state_dict(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"w": torch.zeros(1)}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = _load_weights(str(path), expected_sha256=digest.upper())
    assert set(loaded) == {"w"}
