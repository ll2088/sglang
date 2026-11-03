# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

Upstream Model-Optimizer PR #3063 removed the `weights_only=False`
fallback and required SHA-256 verification for downloaded weights.
These tests exercise the same contract with a local helper so they do
not depend on torch or any heavy runtime.
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


def _safe_load(path, expected_sha256=None):
    """Load a pickle only when its digest matches the expected value."""
    with open(path, "rb") as f:
        data = f.read()
    if expected_sha256 is None:
        raise RuntimeError("Refusing to load pickle without a trusted SHA-256")
    actual = hashlib.sha256(data).hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError("Pickle digest mismatch; refusing to load")
    return pickle.loads(data)


def test_unsafe_payload_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as f:
        pickle.dump(_UnsafePayload(marker), f)
    with pytest.raises(RuntimeError, match="SHA-256"):
        _safe_load(path)
    assert not marker.exists()


def test_digest_mismatch_is_rejected(tmp_path):
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as f:
        pickle.dump({"ok": True}, f)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _safe_load(path, "0" * 64)


def test_matching_digest_loads(tmp_path):
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as f:
        pickle.dump({"ok": True}, f)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _safe_load(path, digest.upper()) == {"ok": True}
