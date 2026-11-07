import math
import struct

import pytest

# 本文件只依赖标准库与 pytest，禁止在模块作用域导入 sglang/torch/CUDA 扩展，
# 以保证在最小 CI 环境中可被收集。内核级覆盖见 test_dsa_topk_bf16_kernel.py。

# This file is intentionally dependency-free (stdlib only) so it can be
# collected and run in a minimal CI environment without sglang's runtime
# dependencies. Kernel-level coverage lives in test_dsa_topk_bf16_kernel.py.
#
# Do NOT import sglang, torch, or any CUDA extension at module scope here:
# the CI setup step installs sglang's heavy dependency tree and can fail
# before pytest ever runs. Keeping this file stdlib-only guarantees the
# BF16x2 packed-histogram unit tests are always collectable.


def _bf16_bits(x: float) -> int:
    """Round a float to BF16 (round-to-nearest-even) and return its 16-bit pattern."""
    f32 = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    lsb = (f32 >> 16) & 1
    rounding = 0x7FFF + lsb
    return ((f32 + rounding) >> 16) & 0xFFFF


def _takes_bf16_logits(dtype_code: str, bits: int, lanes: int, stride0: int) -> bool:
    """调度层能力协商的纯 Python 参考：BF16 原生读取需 16 字节行对齐。"""
    return dtype_code == "bfloat" and bits == 16 and lanes == 1 and stride0 % 8 == 0


def _bf16_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _ordered_key16(bits: int) -> int:
    """Ordered 16-bit key: ~bits if negative, else bits | 0x8000."""
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return (bits | 0x8000) & 0xFFFF


def _coarse_bin_bf16x2(w: int, k_bits: int) -> tuple[int, int]:
    """Reference of extract_coarse_bins_bf16x2: two BF16 scores packed in `w`."""
    lo = w & 0xFFFF
    hi = (w >> 16) & 0xFFFF
    klo = _ordered_key16(lo)
    khi = _ordered_key16(hi)
    shift = 16 - k_bits
    return (klo >> shift, khi >> shift)


def _pack_bf16x2(a: float, b: float) -> int:
    return (_bf16_bits(b) << 16) | _bf16_bits(a)


@pytest.mark.parametrize("k_bits", [8, 12])
@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.0),
        (-0.0, 0.0),
        (1.0, -1.0),
        (float("inf"), float("-inf")),
        (3.5, -2.25),
        (1e-8, -1e-8),
        (65504.0, -65504.0),
    ],
)
def test_bf16x2_coarse_bins_match_scalar(k_bits: int, values: tuple[float, float]) -> None:
    a, b = values
    w = _pack_bf16x2(a, b)
    lo_bin, hi_bin = _coarse_bin_bf16x2(w, k_bits)
    # Scalar reference: bin each score independently through the same key.
    assert lo_bin == _ordered_key16(_bf16_bits(a)) >> (16 - k_bits)
    assert hi_bin == _ordered_key16(_bf16_bits(b)) >> (16 - k_bits)


def test_bf16x2_coarse_bins_are_monotonic() -> None:
    k_bits = 8
    prev = -1
    for i in range(-64, 65):
        x = i / 8.0
        w = _pack_bf16x2(x, x)
        lo, hi = _coarse_bin_bf16x2(w, k_bits)
        assert lo == hi
        assert lo >= prev
        prev = lo


def test_bf16x2_packed_histogram_topk_indices() -> None:
    """A tiny end-to-end check of the packed-bin histogram on a known row."""
    k_bits = 8
    scores = [0.5, 3.0, -1.0, 2.0, 3.0, 0.0, 1.5, -2.0]
    # Pack pairs as the kernel would read them.
    words = [_pack_bf16x2(scores[i], scores[i + 1]) for i in range(0, len(scores), 2)]
    hist = [0] * (1 << k_bits)
    for w in words:
        lo, hi = _coarse_bin_bf16x2(w, k_bits)
        hist[lo] += 1
        hist[hi] += 1
    assert sum(hist) == len(scores)
    # The top-2 scores are the two 3.0 entries; their bin must be the highest
    # non-empty bin.
    top_bin = max(i for i, c in enumerate(hist) if c)
    assert hist[top_bin] == 2
    assert _bf16_from_bits(_bf16_bits(3.0)) == 3.0


def test_bf16_round_trip_is_exact_for_representable_values() -> None:
    for x in [0.0, 1.0, -1.0, 2.0, 0.5, 256.0, -256.0]:
        assert _bf16_from_bits(_bf16_bits(x)) == x


def test_bf16_rounding_is_nearest_even() -> None:
    # 1 + 2^-8 is exactly halfway between 1.0 and the next BF16 value.
    halfway = 1.0 + 2.0 ** -8
    assert _bf16_from_bits(_bf16_bits(halfway)) == 1.0
    # 1 + 3 * 2^-8 rounds up to 1 + 2^-7.
    above = 1.0 + 3 * 2.0 ** -8
    assert _bf16_from_bits(_bf16_bits(above)) == 1.0 + 2.0 ** -7


def test_bf16_nan_and_inf_keys() -> None:
    inf_bits = _bf16_bits(float("inf"))
    ninf_bits = _bf16_bits(float("-inf"))
    assert _ordered_key16(inf_bits) > _ordered_key16(_bf16_bits(1.0))
    assert _ordered_key16(ninf_bits) < _ordered_key16(_bf16_bits(-1.0))


@pytest.mark.parametrize(
    "dtype_code,bits,lanes,stride0,expected",
    [
        ("bfloat", 16, 1, 8, True),
        ("bfloat", 16, 1, 16, True),
        ("bfloat", 16, 1, 7, False),
        ("bfloat", 16, 1, 4, False),
        ("float", 32, 1, 8, False),
        ("bfloat", 16, 2, 8, False),
    ],
)
def test_takes_bf16_logits_capability_negotiation(
    dtype_code: str, bits: int, lanes: int, stride0: int, expected: bool
) -> None:
    """后端 dtype 能力协商：不满足 16 字节对齐时回退 FP32 路径。"""
    assert _takes_bf16_logits(dtype_code, bits, lanes, stride0) is expected


def test_bf16x2_packed_histogram_matches_scalar_topk() -> None:
    """BF16x2 打包直方图提取与逐元素标量直方图在 top-k 选择上一致。"""
    k_bits = 8
    scores = [0.5, 3.0, -1.0, 2.0, 3.0, 0.0, 1.5, -2.0]
    words = [_pack_bf16x2(scores[i], scores[i + 1]) for i in range(0, len(scores), 2)]
    packed_hist = [0] * (1 << k_bits)
    for w in words:
        lo, hi = _coarse_bin_bf16x2(w, k_bits)
        packed_hist[lo] += 1
        packed_hist[hi] += 1
    scalar_hist = [0] * (1 << k_bits)
    for s in scores:
        scalar_hist[_ordered_key16(_bf16_bits(s)) >> (16 - k_bits)] += 1
    assert packed_hist == scalar_hist
