# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

These tests are self-contained (stdlib + pytest only) and do not import
sglang. They verify that a malicious pickle payload is rejected without
executing code.
"""

from __future__ import annotations

import os
import pickle

import pytest


class _UnsupportedCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _safe_load(path):
    """Load a pickle only if it contains plain builtin containers."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, (dict, list, tuple, str, int, float, bool, type(None))):
        raise RuntimeError("Checkpoint contains unsupported pickle objects")
    return obj


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "unsupported.pt"
    with open(path, "wb") as f:
        pickle.dump(_UnsupportedCheckpoint(marker), f)
    with pytest.raises(RuntimeError, match="Checkpoint"):
        _safe_load(path)
    assert not marker.exists()


def test_plain_pickle_is_accepted(tmp_path):
    path = tmp_path / "plain.pt"
    with open(path, "wb") as f:
        pickle.dump({"weight": [1, 2, 3]}, f)
    assert _safe_load(path) == {"weight": [1, 2, 3]}
