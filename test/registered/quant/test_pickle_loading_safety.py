# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe checkpoint loading.

These tests are self-contained: they do not import ``sglang`` so that they can
run without the full runtime dependency tree. They mirror the upstream fix that
removed the ``weights_only=False`` pickle fallback and now requires an explicit
SHA-256 digest for custom weights.
"""

import hashlib
import os

import pytest
import torch


class _UnsupportedCheckpoint:
    """A pickle payload that would execute code if deserialized unsafely."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_weights(path, expected_sha256=None):
    """Mirror the fixed loader: verify digest, then load with weights_only=True."""
    if expected_sha256 is not None:
        actual = _sha256(path)
        if actual.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"Checkpoint digest mismatch: expected {expected_sha256}, got {actual}"
            )
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - surface a controlled error
        raise RuntimeError(f"Checkpoint could not be loaded safely: {exc}") from exc


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "malicious.pt"
    torch.save(_UnsupportedCheckpoint(marker), path)
    with pytest.raises(RuntimeError, match="Checkpoint"):
        _load_weights(path)
    assert not marker.exists()


def test_malicious_pickle_is_rejected_even_with_matching_digest(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "malicious.pt"
    torch.save(_UnsupportedCheckpoint(marker), path)
    with pytest.raises(RuntimeError, match="Checkpoint"):
        _load_weights(path, expected_sha256=_sha256(path))
    assert not marker.exists()


def test_digest_mismatch_is_rejected(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"w": torch.zeros(2)}, path)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _load_weights(path, expected_sha256="0" * 64)


def test_safe_state_dict_loads_with_matching_digest(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"w": torch.arange(4)}, path)
    loaded = _load_weights(path, expected_sha256=_sha256(path).upper())
    assert torch.equal(loaded["w"], torch.arange(4))
