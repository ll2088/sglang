# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

Mirrors the upstream fix: unsafe ``weights_only=False`` fallbacks are removed
and untrusted payloads must be rejected without executing code.
"""

import os
import pickle
from pathlib import Path

import pytest
import torch


class _MaliciousPayload:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _safe_load(path: Path):
    """Load a checkpoint with ``weights_only=True`` only."""
    return torch.load(path, map_location="cpu", weights_only=True)


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    payload_path = tmp_path / "payload.pt"
    with open(payload_path, "wb") as f:
        pickle.dump(_MaliciousPayload(marker), f)

    with pytest.raises(Exception):
        _safe_load(payload_path)

    assert not marker.exists(), "malicious payload executed during load"


def test_trusted_state_dict_roundtrip(tmp_path):
    path = tmp_path / "weights.pt"
    state = {"weight": torch.arange(4, dtype=torch.float32)}
    torch.save(state, path)

    loaded = _safe_load(path)
    assert set(loaded) == set(state)
    torch.testing.assert_close(loaded["weight"], state["weight"])
