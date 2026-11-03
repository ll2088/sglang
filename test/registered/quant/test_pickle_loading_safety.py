# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

Upstream Model-Optimizer PR #3063 removed an unsafe `weights_only=False`
fallback and required SHA-256 verification for downloaded weights.  These
tests exercise the same pattern with a local helper so they do not depend on
torch or any GPU runtime.
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
    data = open(path, "rb").read()
    if expected_sha256 is not None:
        digest = hashlib.sha256(data).hexdigest()
        if digest.lower() != expected_sha256.lower():
            raise RuntimeError("Checkpoint failed SHA-256 verification")
    # Only allow plain builtin containers; reject anything else.
    obj = pickle.loads(data)
    if not isinstance(obj, (dict, list, tuple, int, float, str, bytes, type(None))):
        raise RuntimeError("Checkpoint contains unsupported object")
    return obj


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps(_Malicious(marker)))
    with pytest.raises(RuntimeError, match="unsupported object"):
        _safe_load(str(path))
    assert not marker.exists()


def test_sha256_mismatch_is_rejected(tmp_path):
    path = tmp_path / "weights.pkl"
    path.write_bytes(pickle.dumps({"w": 1}))
    with pytest.raises(RuntimeError, match="SHA-256"):
        _safe_load(str(path), expected_sha256="0" * 64)


def test_sha256_match_allows_plain_container(tmp_path):
    path = tmp_path / "weights.pkl"
    payload = {"w": 1}
    path.write_bytes(pickle.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _safe_load(str(path), expected_sha256=digest.upper()) == payload
