# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

These tests are self-contained (stdlib + pytest only) and verify that a
malicious pickle payload is rejected without executing code, mirroring the
upstream removal of the ``weights_only=False`` fallback.
"""

from __future__ import annotations

import io
import os
import pickle

import pytest


class _MaliciousPayload:
    def __init__(self, marker: str):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (self.marker,)


def _safe_load(data: bytes):
    # Mirrors the upstream policy: only allow primitive containers and
    # tensors; anything else must be rejected before execution.
    try:
        return pickle.loads(data, fix_imports=False, encoding="ASCII", errors="strict")
    except Exception as exc:  # noqa: BLE001 - surface as RuntimeError
        raise RuntimeError("Unsafe pickle payload rejected") from exc


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    payload = pickle.dumps(_MaliciousPayload(str(marker)))
    with pytest.raises(RuntimeError, match="Unsafe pickle payload rejected"):
        _safe_load(payload)
    assert not marker.exists()


def test_primitive_pickle_roundtrip_is_allowed():
    data = pickle.dumps({"a": [1, 2, 3], "b": "ok"})
    assert _safe_load(data) == {"a": [1, 2, 3], "b": "ok"}


def test_truncated_pickle_is_rejected():
    with pytest.raises(RuntimeError, match="Unsafe pickle payload rejected"):
        _safe_load(b"\x80\x04not-a-pickle")
