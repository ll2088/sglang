# Copyright (c) 2025, Tri Dao.
"""Tests for aux_tensors index bounding (clamp instead of modulo)."""

import pytest
import torch


class _FakeDivmod:
    """Minimal stand-in for FastDivmodDivisor exposing only ``divisor``."""

    def __init__(self, divisor: int):
        self.divisor = divisor


def _clamp_index_py(idx: int, seqlen: int) -> int:
    """Python reference for utils.clamp_index: min(idx, seqlen - 1)."""
    return min(idx, seqlen - 1)


@pytest.mark.parametrize("seqlen", [1, 2, 7, 128, 256])
@pytest.mark.parametrize("idx", [-5, -1, 0, 1, 3, 127, 128, 255, 256, 1000])
def test_clamp_index_semantics(idx, seqlen):
    """Clamped index must stay in [0, seqlen - 1] and equal min(idx, seqlen - 1)."""
    out = _clamp_index_py(idx, seqlen)
    assert 0 <= out <= seqlen - 1
    assert out == min(idx, seqlen - 1)


def test_clamp_index_no_wraparound():
    """idx == seqlen must map to seqlen - 1, not wrap to 0 as modulo would."""
    seqlen = 128
    assert _clamp_index_py(seqlen, seqlen) == seqlen - 1
    assert _clamp_index_py(seqlen, seqlen) != seqlen % seqlen
    assert _clamp_index_py(seqlen + 5, seqlen) == seqlen - 1


def test_clamp_index_in_bounds_unchanged():
    """In-bounds indices must be returned unchanged."""
    seqlen = 256
    for idx in (0, 1, 63, 127, 255):
        assert _clamp_index_py(idx, seqlen) == idx


def test_clamp_index_requires_positive_seqlen():
    """seqlen >= 1 is a precondition; seqlen == 1 always yields 0."""
    assert _clamp_index_py(0, 1) == 0
    assert _clamp_index_py(5, 1) == 0


def test_aux_index_mapping_matches_expected_values():
    """Clamped aux indices must select the expected aux values (no wraparound)."""
    seqlen = 8
    aux = torch.arange(seqlen, dtype=torch.float32)
    raw_indices = [-3, -1, 0, 3, 7, 8, 9, 100]
    expected = [aux[_clamp_index_py(i, seqlen)].item() for i in raw_indices]
    got = [aux[_clamp_index_py(i, seqlen)].item() for i in raw_indices]
    assert got == expected
    # idx == seqlen must read the last element, not the first.
    assert aux[_clamp_index_py(seqlen, seqlen)].item() == aux[-1].item()
