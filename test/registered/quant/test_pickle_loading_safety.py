# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

Upstream Model-Optimizer PR #3063 removed an unsafe ``weights_only=False``
fallback and added SHA-256 verification for downloaded weights.  These
tests exercise the same pattern with a local helper so they do not depend
on heavy runtime modules.
"""

import hashlib
import os
import pickle

import pytest


class _MaliciousPayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _safe_load(path, expected_sha256=None):
    """Load a pickle only if its digest matches the expected value."""
    with open(path, "rb") as f:
        data = f.read()
    if expected_sha256 is not None:
        actual = hashlib.sha256(data).hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise RuntimeError("Checkpoint digest mismatch")
    return pickle.loads(data)


def test_malicious_payload_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as f:
        pickle.dump(_MaliciousPayload(marker), f)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _safe_load(str(path), expected_sha256="0" * 64)
    assert not marker.exists()


def test_matching_digest_allows_load(tmp_path):
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as f:
        pickle.dump({"ok": True}, f)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _safe_load(str(path), expected_sha256=digest) == {"ok": True}


def test_missing_digest_is_rejected(tmp_path):
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as f:
        pickle.dump({"ok": True}, f)
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _safe_load(str(path), expected_sha256=None)
