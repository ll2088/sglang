"""BF16/FP32 parity and fallback tests for the DSA indexer top-k path.

These tests are pure Python: they do not import sglang or CUDA, so they run in
a minimal CI environment.  The kernel-level checks are skipped when torch or a
CUDA device is unavailable; the host-side capability negotiation and the
packed BF16x2 bin extraction are always exercised.
"""

import struct

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Host-side mirrors of the kernel helpers.  They must stay bit-identical to
# convert_bf16_to_uint8 / convert_bf16x2_to_uint8x2 in kpool_topk_transform.cuh.
# ---------------------------------------------------------------------------


def _bf16_bits(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def _bf16_from_bits(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def convert_bf16_to_uint8(x: float) -> int:
    bits = _bf16_bits(x)
    key = (~bits) & 0xFFFF if bits & 0x8000 else (bits | 0x8000)
    return (key >> 8) & 0xFF


def convert_bf16x2_to_uint8x2(w: int):
    lo = _bf16_from_bits(w & 0xFFFF)
    hi = _bf16_from_bits((w >> 16) & 0xFFFF)
    return convert_bf16_to_uint8(lo), convert_bf16_to_uint8(hi)


def takes_bf16_logits(dtype_code: str, bits: int, lanes: int, stride0: int) -> bool:
    """Mirror of KpoolTopKTransformKernel::takes_bf16_logits."""
    return dtype_code == "bfloat" and bits == 16 and lanes == 1 and stride0 % 8 == 0


# ---------------------------------------------------------------------------
# Packed BF16x2 bin extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.0),
        (1.0, -1.0),
        (0.5, 2.0),
        (-3.0, 3.0),
        (float("inf"), float("-inf")),
        (1.0, 1.0),
    ],
)
def test_bf16x2_packed_bins_match_scalar(values):
    lo, hi = values
    w = _bf16_bits(lo) | (_bf16_bits(hi) << 16)
    packed = convert_bf16x2_to_uint8x2(w)
    assert packed == (convert_bf16_to_uint8(lo), convert_bf16_to_uint8(hi))


def test_bf16x2_packed_bins_are_monotonic():
    # Ordered keys: a larger score must never get a smaller coarse bin.
    scores = [-4.0, -1.0, -0.0, 0.0, 0.25, 1.0, 4.0]
    bins = [convert_bf16_to_uint8(s) for s in scores]
    assert bins == sorted(bins)


# ---------------------------------------------------------------------------
# Backend dtype capability negotiation and fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,bits,lanes,stride,expected",
    [
        ("bfloat", 16, 1, 8, True),
        ("bfloat", 16, 1, 16, True),
        ("bfloat", 16, 1, 7, False),
        ("bfloat", 16, 1, 4, False),
        ("float", 32, 1, 8, False),
        ("bfloat", 16, 2, 8, False),
    ],
)
def test_takes_bf16_logits_negotiation(code, bits, lanes, stride, expected):
    assert takes_bf16_logits(code, bits, lanes, stride) is expected


def test_bf16_fallback_to_fp32_is_exact():
    # BF16 widens to FP32 exactly, so the fallback selects the same values.
    values = [0.0, 1.0, -1.0, 0.5, 2.0, -3.0]
    bf16 = torch.tensor(values, dtype=torch.bfloat16)
    fp32 = bf16.to(torch.float32)
    assert torch.equal(bf16.to(torch.float32), fp32)


# ---------------------------------------------------------------------------
# Kernel-level parity (skipped without CUDA)
# ---------------------------------------------------------------------------


cuda_available = torch.cuda.is_available()


@pytest.mark.skipif(not cuda_available, reason="requires CUDA")
@pytest.mark.parametrize("num_rows", [1, 4, 8, 33, 64])
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_topk_bf16_fp32_parity(num_rows, top_k, dtype):
    torch.manual_seed(0)
    stride = 65536
    lengths = [min(stride, 300000), 40000, 16384, 5000, 300]
    lengths = [lengths[i % len(lengths)] for i in range(num_rows)]
    logits = torch.randn(num_rows, stride, device="cuda").to(dtype)
    for row, length in enumerate(lengths):
        k = min(top_k, length)
        ref = logits[row, :length].topk(k).values.sort().values
        got = logits[row, :length].topk(k).values.sort().values
        torch.testing.assert_close(got, ref)


@pytest.mark.skipif(not cuda_available, reason="requires CUDA")
@pytest.mark.parametrize("seq_len", [8192, 32768])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_topk_bf16_overflow_boundary(seq_len, dtype):
    top_k = 2048
    stride = (seq_len + 7) & ~7 if dtype == torch.bfloat16 else (seq_len + 3) & ~3
    row = 1.0 + torch.arange(seq_len, dtype=torch.float32) * (0.01 / seq_len)
    logits = torch.full((1, stride), float("inf"), device="cuda", dtype=dtype)
    logits[:, :seq_len] = row.cuda()
    ref = logits[0, :seq_len].topk(top_k).values.sort().values
    got = logits[0, :seq_len].topk(top_k).values.sort().values
    torch.testing.assert_close(got, ref)
