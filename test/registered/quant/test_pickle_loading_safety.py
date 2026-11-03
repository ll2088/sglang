# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

These tests exercise the whitelist / digest checks in
``sglang.srt.utils.safe_pickle`` without importing the full sglang runtime.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import pickle
from pathlib import Path

import pytest

_SAFE_PICKLE = Path(__file__).parents[3] / "python/sglang/srt/utils/safe_pickle.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("safe_pickle_under_test", _SAFE_PICKLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Malicious:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    safe_pickle = _load_module()
    marker = tmp_path / "executed"
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps(_Malicious(marker)))

    with pytest.raises(safe_pickle.UnsafePickleError):
        safe_pickle.safe_pickle_load(str(payload))
    assert not marker.exists()


def test_digest_mismatch_is_rejected(tmp_path):
    safe_pickle = _load_module()
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps({"a": 1}))

    with pytest.raises(safe_pickle.UnsafePickleError):
        safe_pickle.safe_pickle_load(str(payload), expected_sha256="0" * 64)


def test_digest_match_allows_plain_data(tmp_path):
    safe_pickle = _load_module()
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps({"a": 1}))
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()

    assert safe_pickle.safe_pickle_load(str(payload), expected_sha256=digest) == {"a": 1}
