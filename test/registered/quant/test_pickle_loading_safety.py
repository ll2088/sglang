# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

Upstream Model-Optimizer PR #3063 removed an unsafe `weights_only=False`
fallback and added SHA-256 verification for downloaded weights.  These
tests exercise the same pattern with a local loader so they do not depend
on any heavy runtime.
"""

import hashlib
import io
import os
import pickle

import pytest


class _MaliciousPayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _safe_load(data: bytes, expected_sha256: str):
    """Load a pickle only after verifying its SHA-256 digest."""
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256.lower():
        raise RuntimeError("Checkpoint failed SHA-256 verification")
    return pickle.loads(data)


def test_malicious_payload_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    payload = pickle.dumps(_MaliciousPayload(marker))
    with pytest.raises(RuntimeError, match="SHA-256"):
        _safe_load(payload, "0" * 64)
    assert not marker.exists()


def test_trusted_payload_loads_after_verification():
    payload = pickle.dumps({"weight": [1, 2, 3]})
    digest = hashlib.sha256(payload).hexdigest()
    assert _safe_load(payload, digest.upper()) == {"weight": [1, 2, 3]}


def test_modified_payload_fails_verification():
    payload = pickle.dumps({"weight": [1, 2, 3]})
    digest = hashlib.sha256(payload).hexdigest()
    tampered = io.BytesIO()
    tampered.write(payload)
    tampered.write(b"x")
    with pytest.raises(RuntimeError, match="SHA-256"):
        _safe_load(tampered.getvalue(), digest)
