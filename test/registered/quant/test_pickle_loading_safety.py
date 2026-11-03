# SPDX-License-Identifier: Apache-2.0
"""Regression tests ensuring unsafe pickle payloads are rejected."""

import os

import pytest
import torch


class _UnsupportedCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def test_torch_load_weights_only_rejects_arbitrary_pickle(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "unsupported.pt"
    torch.save(_UnsupportedCheckpoint(marker), path)
    with pytest.raises(Exception):
        torch.load(path, map_location="cpu", weights_only=True)
    assert not marker.exists()


def test_torch_load_weights_only_accepts_state_dict(tmp_path):
    path = tmp_path / "weights.pt"
    state = {"weight": torch.zeros(2, 2)}
    torch.save(state, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert set(loaded) == {"weight"}
    assert torch.equal(loaded["weight"], state["weight"])
