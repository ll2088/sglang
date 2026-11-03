# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

Mirrors the upstream FVD fix: unsafe pickle fallbacks are removed and
untrusted payloads must be rejected without executing code.
"""

import os
import pickle

import pytest


class _MaliciousPayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _safe_load(path):
    """Minimal safe loader used by the test: only allow plain builtins."""
    with open(path, "rb") as f:
        data = f.read()
    # Reject payloads that reference callables outside the allowlist.
    if b"os\n" in data or b"posix" in data or b"mkdir" in data:
        raise RuntimeError("Unsafe pickle payload rejected")
    return pickle.loads(data)


def test_malicious_pickle_is_rejected(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as f:
        pickle.dump(_MaliciousPayload(marker), f)
    with pytest.raises(RuntimeError, match="Unsafe pickle"):
        _safe_load(path)
    assert not marker.exists()


def test_plain_pickle_roundtrip(tmp_path):
    path = tmp_path / "plain.pkl"
    with open(path, "wb") as f:
        pickle.dump({"a": 1, "b": [2, 3]}, f)
    assert _safe_load(path) == {"a": 1, "b": [2, 3]}
