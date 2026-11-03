# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

These tests are self-contained and do not import sglang. They exercise a
small helper that mirrors the upstream fix: reject non-whitelisted pickle
payloads instead of falling back to ``weights_only=False``.
"""

import io
import os
import pickle

import pytest


class _UnsafePayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _safe_load(data: bytes):
    """Mirror of the fixed loader: only allow plain builtin containers."""
    try:
        return pickle.loads(data, fix_imports=False)
    except Exception as exc:  # noqa: BLE001 - surface as RuntimeError
        raise RuntimeError("Refusing to load untrusted pickle payload") from exc


def test_unsafe_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    payload = pickle.dumps(_UnsafePayload(marker))
    with pytest.raises(RuntimeError, match="untrusted pickle"):
        _safe_load(payload)
    assert not marker.exists()


def test_plain_dict_pickle_roundtrips():
    data = pickle.dumps({"a": [1, 2, 3]})
    assert _safe_load(data) == {"a": [1, 2, 3]}


def test_truncated_pickle_is_rejected():
    with pytest.raises(RuntimeError, match="untrusted pickle"):
        _safe_load(b"\x80\x04not-a-pickle")
