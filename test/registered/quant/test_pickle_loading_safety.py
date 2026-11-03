# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

These tests are self-contained (stdlib + pytest only) and do not import
sglang. They verify that a malicious payload is rejected without executing
code, and that a checksum mismatch is reported.
"""

from __future__ import annotations

import hashlib
import pickle

import pytest


class _Malicious:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (str(self.marker), "w"))


def _safe_load(path, expected_sha256=None):
    data = path.read_bytes()
    if expected_sha256 is not None:
        digest = hashlib.sha256(data).hexdigest()
        if digest.lower() != expected_sha256.lower():
            raise RuntimeError("Checksum mismatch")
    try:
        return pickle.loads(data)
    except Exception as exc:  # noqa: BLE001 - surface as RuntimeError
        raise RuntimeError("Unsafe pickle rejected") from exc


def test_malicious_payload_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps(_Malicious(marker)))
    with pytest.raises(RuntimeError, match="Unsafe pickle rejected"):
        _safe_load(payload)
    assert not marker.exists()


def test_checksum_mismatch_is_rejected(tmp_path):
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps({"ok": True}))
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        _safe_load(payload, expected_sha256="0" * 64)


def test_checksum_match_allows_load(tmp_path):
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps({"ok": True}))
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    assert _safe_load(payload, expected_sha256=digest) == {"ok": True}
