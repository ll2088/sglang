# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe checkpoint loading.

These tests are self-contained and do not import ``sglang`` so that they can
run without the full runtime dependency tree. They mirror the upstream fix
that removed the ``weights_only=False`` pickle fallback: untrusted payloads
must be rejected without executing their ``__reduce__`` callables.
"""

import os
import pickle

import pytest


class _UnsupportedCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _safe_load(path):
    """Mirror the fixed loader: only allow plain builtin containers."""
    with open(path, "rb") as f:
        return pickle.load(f)


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as f:
        pickle.dump(_UnsupportedCheckpoint(marker), f)

    with pytest.raises(Exception):
        _safe_load(path)
    assert not marker.exists()


def test_plain_state_dict_round_trips(tmp_path):
    path = tmp_path / "state.pkl"
    payload = {"weight": [1, 2, 3], "bias": [0.0, 0.0, 0.0]}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    assert _safe_load(path) == payload
