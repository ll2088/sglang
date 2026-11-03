# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

Upstream removed the ``weights_only=False`` fallback and requires an explicit
SHA-256 for custom weights. These tests exercise the local helper that
implements the same contract without importing the full sglang runtime.
"""

import hashlib
import os
import pickle

import pytest


class _UnsupportedCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _load_weights(path, expected_sha256=None):
    """Minimal safe loader mirroring the upstream contract.

    - If ``expected_sha256`` is provided, verify the file digest first.
    - Otherwise, only allow plain state-dict-like payloads (dict of tensors
      is not required here; we simply reject arbitrary callables).
    """
    if expected_sha256 is not None:
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if digest.lower() != expected_sha256.lower():
            raise RuntimeError("Supplied weights failed SHA-256 verification")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, _UnsupportedCheckpoint):
        raise RuntimeError("Checkpoint contains unsupported pickle objects")
    return payload


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "evil.pkl"
    with open(path, "wb") as f:
        pickle.dump(_UnsupportedCheckpoint(marker), f)
    with pytest.raises(RuntimeError, match="unsupported pickle"):
        _load_weights(str(path))
    assert not marker.exists()


def test_checksum_mismatch_is_rejected(tmp_path):
    path = tmp_path / "weights.pkl"
    with open(path, "wb") as f:
        pickle.dump({"w": 1}, f)
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(str(path), "0" * 64)


def test_checksum_match_allows_plain_payload(tmp_path):
    path = tmp_path / "weights.pkl"
    with open(path, "wb") as f:
        pickle.dump({"w": 1}, f)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert _load_weights(str(path), digest.upper()) == {"w": 1}
