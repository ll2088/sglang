# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading of model weights.

These tests exercise the shared safe-loading helper used by model weight
loaders. They assert that untrusted payloads are rejected without executing
any code, and that a digest mismatch is reported as a controlled error.
"""

import hashlib
import os

import pytest
import torch

from sglang.srt.utils.safe_pickle import (
    UnsafeCheckpointError,
    load_checkpoint_safely,
    verify_sha256,
)


class _UnsupportedCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "malicious.pt"
    torch.save(_UnsupportedCheckpoint(marker), path)

    with pytest.raises(UnsafeCheckpointError):
        load_checkpoint_safely(str(path), map_location="cpu")

    assert not marker.exists()


def test_state_dict_roundtrip_is_allowed(tmp_path):
    path = tmp_path / "weights.pt"
    state = {"w": torch.arange(4, dtype=torch.float32)}
    torch.save(state, path)

    loaded = load_checkpoint_safely(str(path), map_location="cpu")
    assert set(loaded) == {"w"}
    torch.testing.assert_close(loaded["w"], state["w"])


def test_verify_sha256_accepts_matching_digest(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"trusted checkpoint")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert verify_sha256(str(path), digest) is True
    assert verify_sha256(str(path), digest.upper()) is True


def test_verify_sha256_rejects_mismatch(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"trusted checkpoint")

    with pytest.raises(UnsafeCheckpointError):
        verify_sha256(str(path), "0" * 64)


def test_verify_sha256_requires_digest(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"trusted checkpoint")

    with pytest.raises(UnsafeCheckpointError):
        verify_sha256(str(path), None)
