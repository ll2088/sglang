# SPDX-License-Identifier: Apache-2.0
"""Security regression tests for safe pickle loading.

Only the standard library and pytest are required; the module under test is
imported directly from its file path so that importing sglang is not needed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import pickle
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sglang"
    / "srt"
    / "utils"
    / "safe_pickle.py"
)
_SPEC = importlib.util.spec_from_file_location("safe_pickle_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
safe_pickle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(safe_pickle)

UnsafePickleError = safe_pickle.UnsafePickleError
safe_pickle_load = safe_pickle.safe_pickle_load
safe_pickle_loads = safe_pickle.safe_pickle_loads
sha256_file = safe_pickle.sha256_file


class _Malicious:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def test_malicious_payload_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    payload = pickle.dumps(_Malicious(marker))
    with pytest.raises(UnsafePickleError):
        safe_pickle_loads(payload)
    assert not marker.exists()


def test_allow_listed_payload_round_trips():
    payload = pickle.dumps({"a": [1, 2, 3], "b": ("x", "y")})
    assert safe_pickle_loads(payload) == {"a": [1, 2, 3], "b": ("x", "y")}


def test_load_requires_explicit_digest(tmp_path):
    path = tmp_path / "data.pkl"
    path.write_bytes(pickle.dumps({"k": 1}))
    with pytest.raises(UnsafePickleError, match="SHA-256"):
        safe_pickle_load(path)


def test_load_rejects_wrong_digest(tmp_path):
    path = tmp_path / "data.pkl"
    path.write_bytes(pickle.dumps({"k": 1}))
    with pytest.raises(UnsafePickleError, match="failed SHA-256"):
        safe_pickle_load(path, expected_sha256="0" * 64)


def test_load_accepts_matching_digest(tmp_path):
    path = tmp_path / "data.pkl"
    path.write_bytes(pickle.dumps({"k": 1}))
    digest = sha256_file(path)
    assert safe_pickle_load(path, expected_sha256=digest.upper()) == {"k": 1}


def test_load_rejects_malicious_payload_even_with_valid_digest(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps(_Malicious(marker)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(UnsafePickleError):
        safe_pickle_load(path, expected_sha256=digest)
    assert not marker.exists()
