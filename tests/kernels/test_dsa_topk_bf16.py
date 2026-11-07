"""Minimal-dependency tests for the DSA indexer top-k BF16 capability.

These tests only need numpy and pytest.  They exercise the pure-Python
negotiation helpers and the host-side mirror of the BF16x2 packed histogram
extraction, so CI can run them without installing the full sglang runtime.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest


def _load_capability_module():
    """Load bf16_capability.py by path so this test never imports sglang.

    Importing ``sglang.kernels.jit.csrc.dsa`` pulls in the package __init__
    chain, which needs torch and the full runtime.  The capability helper is a
    standalone pure-Python module, so load it directly from its file.

    This module must stay importable with only numpy + pytest installed; see
    tests/kernels/requirements-minimal.txt.  If the helper is missing we skip
    with an explicit reason instead of letting the import chain pull in the
    full sglang dependency set.
    """
    here = pathlib.Path(__file__).resolve()
    repo_root = here.parents[2]
    path = repo_root / "python" / "sglang" / "kernels" / "jit" / "csrc" / "dsa" / "bf16_capability.py"
    if not path.is_file():
        pytest.skip(
            f"bf16_capability.py not found at {path}; "
            "this test only needs numpy+pytest (tests/kernels/requirements-minimal.txt)"
        )
    spec = importlib.util.spec_from_file_location("dsa_bf16_capability", path)
    if spec is None or spec.loader is None:
        pytest.skip(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cap = _load_capability_module()


# ---------------------------------------------------------------------------
# Host-side mirror of convert_bf16x2_to_uint8x2 (see kpool_topk_transform.cuh).
# ---------------------------------------------------------------------------


def _host_convert_bf16_to_uint8(bits: int) -> int:
    bits &= 0xFFFF
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def _host_convert_bf16x2_to_uint8x2(w: int) -> tuple[int, int]:
    lo = w & 0xFFFF
    hi = (w >> 16) & 0xFFFF
    return _host_convert_bf16_to_uint8(lo), _host_convert_bf16_to_uint8(hi)


def _bf16_bits(x: float) -> int:
    # Round-to-nearest-even BF16 of a float32 value.
    f = np.float32(x)
    u = int(np.frombuffer(f.tobytes(), dtype=np.uint32)[0])
    # round to nearest even on the low 16 bits
    lsb = (u >> 16) & 1
    rounding = 0x7FFF + lsb
    u = (u + rounding) & 0xFFFFFFFF
    return (u >> 16) & 0xFFFF


def _bf16_to_float32(bits: int) -> np.float32:
    """Widen a BF16 bit pattern to float32 exactly (BF16 is the high half)."""
    raw = np.uint32((bits & 0xFFFF) << 16).tobytes()
    return np.frombuffer(raw, dtype=np.float32)[0]


# ---------------------------------------------------------------------------
# Capability negotiation
# ---------------------------------------------------------------------------


def test_vec_width_matches_element_width():
    assert cap.vec_width(4) == 4
    assert cap.vec_width(2) == 8
    assert cap.tma_align(4) == 4
    assert cap.tma_align(2) == 8


def test_vec_width_rejects_odd_width():
    with pytest.raises(ValueError):
        cap.vec_width(3)


@pytest.mark.parametrize("stride", [8, 16, 64, 65536, 1048576])
def test_bf16_taken_when_aligned(stride):
    assert cap.takes_bf16_logits(cap.BF16, stride)
    assert cap.resolve_logits_dtype(cap.BF16, stride) == "bf16"


@pytest.mark.parametrize("stride", [1, 2, 3, 4, 7, 9, 15])
def test_bf16_falls_back_when_unaligned(stride):
    assert not cap.takes_bf16_logits(cap.BF16, stride)
    assert cap.resolve_logits_dtype(cap.BF16, stride) == "fp32"


def test_fp32_always_taken():
    assert cap.takes_fp32_logits(cap.FP32)
    assert not cap.takes_bf16_logits(cap.FP32, 8)
    assert cap.resolve_logits_dtype(cap.FP32, 8) == "fp32"


def test_bf16_rejects_non_positive_stride():
    assert not cap.takes_bf16_logits(cap.BF16, 0)
    assert not cap.takes_bf16_logits(cap.BF16, -8)


# ---------------------------------------------------------------------------
# BF16x2 packed histogram extraction
# ---------------------------------------------------------------------------


def test_bf16x2_matches_scalar_extraction():
    rng = np.random.default_rng(0)
    for _ in range(256):
        a = float(rng.standard_normal())
        b = float(rng.standard_normal())
        w = _bf16_bits(a) | (_bf16_bits(b) << 16)
        lo, hi = _host_convert_bf16x2_to_uint8x2(w)
        assert lo == _host_convert_bf16_to_uint8(_bf16_bits(a))
        assert hi == _host_convert_bf16_to_uint8(_bf16_bits(b))


def test_bf16x2_ordering_is_monotonic():
    # The ordered key must be monotonic in the score.
    vals = np.linspace(-4.0, 4.0, 257)
    keys = [_host_convert_bf16_to_uint8(_bf16_bits(float(v))) for v in vals]
    assert keys == sorted(keys)


def test_bf16x2_handles_signed_zero():
    pos = _bf16_bits(0.0)
    neg = _bf16_bits(-0.0)
    assert _host_convert_bf16_to_uint8(pos) == _host_convert_bf16_to_uint8(neg)


def test_bf16x2_topk_indices_match_fp32_reference():
    rng = np.random.default_rng(1)
    n = 4096
    k = 64
    scores = rng.standard_normal(n).astype(np.float32)
    # BF16-rounded scores, then top-k on the rounded values.
    bf16 = np.array([_bf16_to_float32(_bf16_bits(float(s))) for s in scores], dtype=np.float32)
    ref = np.argsort(-bf16, kind="stable")[:k]
    got = np.argsort(-scores, kind="stable")[:k]
    # The selected values must agree even if tie-breaking differs.
    np.testing.assert_allclose(np.sort(bf16[ref]), np.sort(scores[got]), rtol=0, atol=0)
