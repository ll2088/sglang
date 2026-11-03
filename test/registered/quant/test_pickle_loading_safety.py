# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

Mirrors the upstream fix pattern: untrusted checkpoints must be loaded with
``weights_only=True`` (or an explicit digest check) so that malicious pickle
payloads are rejected instead of executed.
"""

import hashlib
import os

import pytest
import torch


class _UnsupportedCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _load_weights_only(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_with_digest(path, expected_sha256):
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if digest.lower() != expected_sha256.lower():
        raise RuntimeError("Checkpoint digest mismatch")
    return torch.load(path, map_location="cpu", weights_only=True)


def test_weights_only_rejects_malicious_payload(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "malicious.pt"
    torch.save(_UnsupportedCheckpoint(marker), path)
    with pytest.raises(Exception):
        _load_weights_only(path)
    assert not marker.exists()


def test_digest_mismatch_is_rejected(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"w": torch.zeros(2)}, path)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _load_with_digest(path, "0" * 64)


def test_digest_match_loads_state_dict(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"w": torch.zeros(2)}, path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    loaded = _load_with_digest(path, digest.upper())
    assert torch.equal(loaded["w"], torch.zeros(2))
