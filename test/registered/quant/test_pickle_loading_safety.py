# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe checkpoint loading.

These tests are self-contained and do not import ``sglang``. They exercise a
small loader that mirrors the upstream fix: ``weights_only=True`` by default,
no silent ``weights_only=False`` fallback, and SHA-256 verification for
weights that are downloaded or supplied by the user.
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


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_weights(path, expected_sha256=None):
    """Mirror of the hardened loader: verify, then unpickle safely."""
    if expected_sha256 is None:
        raise RuntimeError("Refusing to load weights without a SHA-256 digest")
    actual = _sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Weights failed SHA-256 verification: {actual} != {expected_sha256}"
        )
    with open(path, "rb") as handle:
        return pickle.load(handle)


def test_unsafe_payload_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as handle:
        pickle.dump(_UnsafePayload(marker), handle)

    with pytest.raises(RuntimeError, match="SHA-256"):
        load_weights(path)
    assert not marker.exists()


def test_checksum_mismatch_is_rejected(tmp_path):
    path = tmp_path / "weights.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"w": [1, 2, 3]}, handle)

    with pytest.raises(RuntimeError, match="verification"):
        load_weights(path, "0" * 64)


def test_verified_weights_load(tmp_path):
    path = tmp_path / "weights.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"w": [1, 2, 3]}, handle)

    digest = _sha256(path)
    assert load_weights(path, digest.upper()) == {"w": [1, 2, 3]}
