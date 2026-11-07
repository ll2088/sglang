"""BF16 vs FP32 capability negotiation and packed-histogram tests.

These tests are pure Python: they load the capability module by explicit path
so that importing them never triggers ``sglang/__init__.py`` (and therefore the
full runtime dependency resolution).  They are the minimal-env counterpart of
the GPU kernel tests and cover:

* BF16 native-read negotiation and FP32 fallback,
* alignment granularity derived from the element width,
* BF16x2 packed histogram extraction vs the per-element reference,
* top-k selection consistency between the BF16 and FP32 paths.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relpath: str):
    path = _REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cap = _load_module(
    "_dsa_topk_bf16_capability",
    "python/sglang/kernels/dsa_topk_bf16_capability.py",
)


# ---------------------------------------------------------------------------
# dtype / alignment negotiation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stride", [8, 16, 1024, 65536])
def test_bf16_native_read_when_aligned(stride: int) -> None:
    assert cap.takes_bf16_logits_for(cap.KDL_BFLOAT, 16, 1, stride)
    assert cap.resolve_logits_dtype(cap.KDL_BFLOAT, 16, 1, stride) == "bf16"


@pytest.mark.parametrize("stride", [1, 3, 7, 9, 65535])
def test_bf16_falls_back_when_unaligned(stride: int) -> None:
    assert not cap.takes_bf16_logits_for(cap.KDL_BFLOAT, 16, 1, stride)
    assert cap.resolve_logits_dtype(cap.KDL_BFLOAT, 16, 1, stride) == "fp32"


def test_fp32_always_readable() -> None:
    assert cap.takes_fp32_logits_for(cap.KDL_FLOAT, 32, 1)
    assert cap.resolve_logits_dtype(cap.KDL_FLOAT, 32, 1, 3) == "fp32"


def test_unknown_dtype_falls_back_to_fp32() -> None:
    # e.g. fp16 (code 2, bits 16) is not natively read by these kernels.
    assert cap.resolve_logits_dtype(cap.KDL_FLOAT, 16, 1, 8) == "fp32"


def test_vec_width_is_derived_from_element_width() -> None:
    assert cap.vec_width(4) == 4
    assert cap.vec_width(2) == 8
    with pytest.raises(ValueError):
        cap.vec_width(3)


# ---------------------------------------------------------------------------
# BF16x2 packed histogram extraction
# ---------------------------------------------------------------------------


def _bf16_bits(value: float) -> int:
    """Round-to-nearest-even BF16 bit pattern of a float32 value."""
    f32 = np.float32(value)
    bits = int(np.frombuffer(f32.tobytes(), dtype=np.uint32)[0])
    # RNE: add 0x7FFF + lsb before truncating.
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return (rounded >> 16) & 0xFFFF


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.0),
        (1.0, -1.0),
        (0.5, 2.0),
        (-3.25, 7.5),
        (float("inf"), float("-inf")),
    ],
)
def test_bf16x2_packed_matches_per_element(values) -> None:
    lo_bits = _bf16_bits(values[0])
    hi_bits = _bf16_bits(values[1])
    word = cap.pack_bf16x2(lo_bits, hi_bits)
    packed = cap.host_convert_bf16x2_to_uint8x2(word)
    assert packed == (
        cap.host_convert_bf16_to_uint8(lo_bits),
        cap.host_convert_bf16_to_uint8(hi_bits),
    )


def test_bf16x2_packed_is_monotonic() -> None:
    # Ordered keys must be monotonic in the value, which is what makes the
    # histogram threshold sound.
    ordered = sorted([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    bins = [cap.host_convert_bf16_to_uint8(_bf16_bits(v)) for v in ordered]
    assert bins == sorted(bins)


# ---------------------------------------------------------------------------
# top-k selection consistency (BF16 vs FP32)
# ---------------------------------------------------------------------------


def _topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    # Stable, deterministic top-k: sort by (-value, index).
    order = np.lexsort((np.arange(values.size), -values))
    return order[:k]


@pytest.mark.parametrize("length", [300, 5000, 16384, 40000])
@pytest.mark.parametrize("k", [512, 2048])
def test_bf16_and_fp32_select_same_values(length: int, k: int) -> None:
    rng = np.random.default_rng(0)
    fp32 = rng.standard_normal(length).astype(np.float32)
    # BF16 is a strict rounding of FP32, so the selected *values* must match.
    bf16 = fp32.astype(np.float32)
    bf16_bits = np.array([_bf16_bits(v) for v in bf16], dtype=np.uint32)
    bf16_round = np.array(
        [np.frombuffer(np.uint32(b << 16).tobytes(), dtype=np.float32)[0] for b in bf16_bits],
        dtype=np.float32,
    )

    kk = min(k, length)
    idx_fp32 = _topk_indices(fp32, kk)
    idx_bf16 = _topk_indices(bf16_round, kk)
    np.testing.assert_allclose(
        np.sort(fp32[idx_fp32]),
        np.sort(bf16_round[idx_bf16]),
        rtol=0,
        atol=0,
    )


def test_bf16_short_row_selects_everything() -> None:
    length = 300
    k = 512
    values = np.arange(length, dtype=np.float32)
    idx = _topk_indices(values, min(k, length))
    assert idx.size == length
    assert np.array_equal(np.sort(idx), np.arange(length))
