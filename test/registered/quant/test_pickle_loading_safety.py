# SPDX-License-Identifier: Apache-2.0
"""Security regression tests for pickle loading.

These tests only use the standard library so they can run without the
sglang runtime dependency tree.
"""

from __future__ import annotations

import os
import pickle

import pytest


class _MaliciousPayload:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (self.marker,)


def _safe_load(path: str):
    """Load a pickle only if it contains plain builtin containers."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, (dict, list, tuple, str, int, float, bool, type(None))):
        raise RuntimeError("Refusing to load non-primitive pickle payload")
    return data


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as f:
        pickle.dump(_MaliciousPayload(str(marker)), f)
    with pytest.raises(RuntimeError, match="Refusing"):
        _safe_load(str(path))
    assert not marker.exists()


def test_plain_pickle_is_accepted(tmp_path):
    path = tmp_path / "plain.pkl"
    with open(path, "wb") as f:
        pickle.dump({"a": [1, 2, 3]}, f)
    assert _safe_load(str(path)) == {"a": [1, 2, 3]}
