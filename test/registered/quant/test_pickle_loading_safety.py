# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

Upstream Model-Optimizer PR #3063 removed the `weights_only=False` fallback
and required SHA-256 verification for downloaded/cached weights.  These tests
use a local helper mirroring that contract so they do not depend on torch
version specifics.
"""

import hashlib
import os
import pickle

import pytest


class _Malicious:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _safe_load(path, expected_sha256=None):
    """Load a pickle only after verifying its digest."""
    if expected_sha256 is None:
        raise RuntimeError("Refusing to load pickle without a trusted SHA-256")
    data = open(path, "rb").read()
    actual = hashlib.sha256(data).hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError("Pickle digest mismatch")
    return pickle.loads(data)


def test_malicious_payload_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps(_Malicious(marker)))
    with pytest.raises(RuntimeError, match="trusted SHA-256"):
        _safe_load(str(path))
    assert not marker.exists()


def test_digest_mismatch_is_rejected(tmp_path):
    path = tmp_path / "payload.pkl"
    path.write_bytes(pickle.dumps({"a": 1}))
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _safe_load(str(path), "0" * 64)


def test_matching_digest_loads_payload(tmp_path):
    path = tmp_path / "payload.pkl"
    data = pickle.dumps({"a": 1})
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    assert _safe_load(str(path), digest.upper()) == {"a": 1}
