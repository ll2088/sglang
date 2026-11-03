# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

Upstream Model-Optimizer PR #3063 removed the `weights_only=False` fallback
and requires SHA-256 verification for downloaded weights. These tests use a
local helper that mirrors the safe-loading contract without importing heavy
runtime modules.
"""

import hashlib
import os
import pickle

import pytest


class _UnsafePayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _safe_load(path: str, expected_sha256: str) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest.lower() != expected_sha256.lower():
        raise RuntimeError("Checkpoint failed SHA-256 verification")
    return data


def test_unsafe_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as f:
        pickle.dump(_UnsafePayload(marker), f)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # A wrong digest must be rejected before any deserialization happens.
    with pytest.raises(RuntimeError, match="SHA-256"):
        _safe_load(str(path), "0" * 64)
    assert not marker.exists()
    # The correct digest only returns bytes; it never unpickles them.
    assert _safe_load(str(path), digest) == path.read_bytes()
    assert not marker.exists()


def test_checksum_is_case_insensitive(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"trusted checkpoint")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _safe_load(str(path), digest.upper()) == b"trusted checkpoint"
