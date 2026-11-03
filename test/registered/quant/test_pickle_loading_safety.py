# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

These tests are self-contained (stdlib + pytest only) and mirror the upstream
fix that removed the ``weights_only=False`` fallback and requires an explicit
SHA-256 digest for custom checkpoints.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
from pathlib import Path
from typing import Optional

import pytest


class _UnsupportedCheckpoint:
    """A payload whose unpickling would execute ``os.mkdir``."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_weights(path: Path, expected_sha256: Optional[str]) -> bytes:
    """Safe loader: verify digest, then unpickle with a restricted unpickler.

    This mirrors the upstream contract: a custom checkpoint must supply an
    explicit digest, and unpickling must never execute arbitrary callables.
    """
    if expected_sha256 is None:
        raise RuntimeError("Custom checkpoint requires an explicit SHA-256 digest")
    actual = _sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError("Checkpoint failed SHA-256 verification")

    class _RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:  # type: ignore[override]
            raise pickle.UnpicklingError(
                f"Refusing to load {module}.{name} from checkpoint"
            )

    with path.open("rb") as fh:
        return _RestrictedUnpickler(fh).load()


from typing import Any  # noqa: E402  (used by the restricted unpickler)


def test_malicious_payload_is_rejected_without_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps(_UnsupportedCheckpoint(marker)))

    with pytest.raises((RuntimeError, pickle.UnpicklingError)):
        _load_weights(payload, _sha256(payload))

    assert not marker.exists()


def test_custom_checkpoint_requires_digest(tmp_path: Path) -> None:
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps({"weight": 1}))

    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(payload, None)


def test_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps({"weight": 1}))

    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_weights(payload, "0" * 64)


def test_digest_is_case_insensitive(tmp_path: Path) -> None:
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps({"weight": 1}))

    with pytest.raises(pickle.UnpicklingError):
        _load_weights(payload, _sha256(payload).upper())
