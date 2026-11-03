# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

These tests exercise a small, self-contained loader that mirrors the upstream
fix: no `weights_only=False` fallback, and untrusted payloads are rejected
without executing code. They do not import sglang.
"""

from __future__ import annotations

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
    """Load a pickle only if it is trusted.

    - If `expected_sha256` is provided, the file digest must match.
    - Otherwise the payload is rejected: there is no unsafe fallback.
    """
    if expected_sha256 is None:
        raise RuntimeError(
            "Refusing to load untrusted pickle: an explicit SHA-256 is required"
        )
    with open(path, "rb") as f:
        data = f.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError("Pickle checksum mismatch")
    return pickle.loads(data)


def test_untrusted_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as f:
        pickle.dump(_UnsafePayload(marker), f)
    with pytest.raises(RuntimeError, match="untrusted pickle"):
        _safe_load(str(path))
    assert not marker.exists()


def test_checksum_mismatch_is_rejected(tmp_path):
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as f:
        pickle.dump({"ok": True}, f)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _safe_load(str(path), "0" * 64)


def test_trusted_pickle_loads(tmp_path):
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as f:
        pickle.dump({"ok": True}, f)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _safe_load(str(path), digest.upper()) == {"ok": True}
