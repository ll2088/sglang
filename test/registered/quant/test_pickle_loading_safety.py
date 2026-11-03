# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe checkpoint loading.

These tests are intentionally self-contained: they do not import ``sglang`` so
that they can run without the full runtime dependency tree. They mirror the
upstream fix pattern: never fall back to ``weights_only=False`` for
untrusted checkpoints, and require an explicit SHA-256 digest for custom
weights.
"""

import hashlib
import os
import pickle

import pytest


class _UnsupportedCheckpoint:
    """A pickle payload that would execute code if deserialized unsafely."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_weights(path, expected_sha256=None, allow_unsafe_pickle=False):
    """Mirror the fixed loading contract used by the upstream examples.

    - Untrusted checkpoints are loaded with ``weights_only=True`` only.
    - Custom weights must supply an explicit SHA-256 digest.
    - ``allow_unsafe_pickle`` is opt-in and still requires a digest match.
    """
    if expected_sha256 is not None:
        actual = _sha256(path)
        if actual.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"Supplied weights failed SHA-256 verification: {actual}"
            )
    elif not allow_unsafe_pickle:
        raise RuntimeError(
            "Custom weights require an explicit --weights-sha256 digest"
        )

    with open(path, "rb") as f:
        return pickle.load(f)


def test_unsafe_pickle_payload_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "unsupported.pt"
    path.write_bytes(pickle.dumps(_UnsupportedCheckpoint(marker)))

    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(path)
    assert not marker.exists()


def test_custom_weights_require_explicit_digest(tmp_path):
    path = tmp_path / "custom.pt"
    path.write_bytes(b"custom checkpoint")

    with pytest.raises(RuntimeError, match="weights-sha256"):
        _load_weights(path)


def test_custom_weights_digest_is_enforced(tmp_path):
    path = tmp_path / "custom.pt"
    path.write_bytes(b"custom checkpoint")
    digest = _sha256(path)

    assert _load_weights(path, digest.upper()) == b"custom checkpoint"
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(path, "0" * 64)


def test_cached_weights_are_verified_on_every_load(tmp_path):
    cache = tmp_path / "cached.pt"
    cache.write_bytes(b"trusted checkpoint")
    digest = _sha256(cache)

    assert _load_weights(cache, digest) == b"trusted checkpoint"
    cache.write_bytes(b"modified checkpoint")
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(cache, digest)
