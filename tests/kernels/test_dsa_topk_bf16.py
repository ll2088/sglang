import struct

import pytest
import torch


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _has_sgl_kernel() -> bool:
    try:
        import sgl_kernel  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_has_cuda() and _has_sgl_kernel()),
    reason="requires CUDA and sgl_kernel extension; skipped in CPU-only CI",
)


# ---------------------------------------------------------------------------
# Pure-Python reference for the BF16x2 packed coarse-bin extraction.
#
# The kernel path (convert_bf16x2_to_uint8x2 / extract_coarse_bins_bf16x2)
# takes two BF16 scores packed in one uint32 (low half first), converts them
# with one packed FP16 conversion, then flips the sign bit per half to build an
# ordered key.  The reference below mirrors that bit-for-bit so the semantics
# can be checked on CPU without compiling the extension.
# ---------------------------------------------------------------------------


def _bf16_bits_to_float(bits: int) -> float:
    # BF16 is the top 16 bits of an FP32.
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _float_to_bf16_bits(x: float) -> int:
    raw = struct.unpack("<I", struct.pack("<f", x))[0]
    # Round-to-nearest-even on the low 16 bits.
    lsb = (raw >> 16) & 1
    rounding = 0x7FFF + lsb
    return ((raw + rounding) >> 16) & 0xFFFF


def _ordered_key16(bits: int) -> int:
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return (bits | 0x8000) & 0xFFFF


def _packed_bins_bf16x2(w: int, k_bits: int) -> tuple[int, int]:
    lo = w & 0xFFFF
    hi = (w >> 16) & 0xFFFF
    klo = _ordered_key16(lo)
    khi = _ordered_key16(hi)
    shift = 16 - k_bits
    return (klo >> shift, khi >> shift)


def _pack_two_bf16(a: float, b: float) -> int:
    return (_float_to_bf16_bits(a) & 0xFFFF) | ((_float_to_bf16_bits(b) & 0xFFFF) << 16)


@pytest.mark.parametrize("k_bits", [8, 12])
@pytest.mark.parametrize(
    "pair",
    [
        (0.0, 0.0),
        (1.0, -1.0),
        (0.5, 2.0),
        (-3.25, 7.5),
        (float("inf"), float("-inf")),
        (1e-30, -1e-30),
    ],
)
def test_bf16x2_packed_bins_match_scalar(k_bits: int, pair: tuple[float, float]) -> None:
    a, b = pair
    w = _pack_two_bf16(a, b)
    lo_bin, hi_bin = _packed_bins_bf16x2(w, k_bits)

    # Scalar reference: widen BF16 to FP32 exactly, then take the ordered key.
    a_bits = _float_to_bf16_bits(a)
    b_bits = _float_to_bf16_bits(b)
    shift = 16 - k_bits
    assert lo_bin == (_ordered_key16(a_bits) >> shift)
    assert hi_bin == (_ordered_key16(b_bits) >> shift)


def test_bf16x2_packed_bins_are_monotonic() -> None:
    values = [-4.0, -1.0, -0.5, 0.0, 0.5, 1.0, 4.0]
    k_bits = 8
    prev = -1
    for v in values:
        w = _pack_two_bf16(v, v)
        lo, hi = _packed_bins_bf16x2(w, k_bits)
        assert lo == hi
        assert lo >= prev
        prev = lo


# ---------------------------------------------------------------------------
# GPU tests.  These are skipped unless CUDA and the sgl_kernel extension are
# both available; they are the ones that exercise the real kernels.
# ---------------------------------------------------------------------------


def _reference_topk(logits: torch.Tensor, lengths: torch.Tensor, k: int):
    out = []
    for row, length in enumerate(lengths.tolist()):
        vals, idx = logits[row, :length].topk(min(k, length))
        out.append((vals, idx))
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_rows", [1, 4, 8, 33, 64])
@pytest.mark.parametrize("stride", [65536, 1048576])
@pytest.mark.parametrize("top_k", [512, 2048])
@torch.inference_mode()
def test_cooperative_topk_bf16_matches_fp32(
    dtype: torch.dtype, num_rows: int, stride: int, top_k: int
) -> None:
    torch.manual_seed(0)
    device = "cuda:0"
    row_lengths = [min(stride, 300000), 40000, 16384, 5000, 300]
    lengths = torch.tensor(
        [row_lengths[i % len(row_lengths)] for i in range(num_rows)],
        dtype=torch.int32,
        device=device,
    )
    logits = torch.randn(num_rows, stride, device=device).to(dtype)
    out = torch.empty(num_rows, top_k, dtype=torch.int32, device=device)

    torch.ops._C.cooperative_topk(logits, lengths, out, top_k)

    ref = _reference_topk(logits.float(), lengths.cpu(), top_k)
    for row, (vals, _) in enumerate(ref):
        k = vals.numel()
        idx = out[row, :k].long()
        assert idx.unique().numel() == k
        torch.testing.assert_close(
            logits[row, idx].float().sort().values,
            vals.sort().values,
        )
        assert torch.all(out[row, k:] == -1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_rows", [9, 34, 64])
@pytest.mark.parametrize("seq_len", [65535, 65536, 65537])
@torch.inference_mode()
def test_cooperative_topk_overflow_cluster_boundary(
    dtype: torch.dtype, num_rows: int, seq_len: int
) -> None:
    top_k = 2048
    align = 8 if dtype == torch.bfloat16 else 4
    stride = (seq_len + align - 1) & ~(align - 1)
    row = 1.0 + torch.arange(seq_len, dtype=torch.float32) * (0.01 / seq_len)
    logits = torch.full((num_rows, stride), float("inf"), device="cuda", dtype=dtype)
    logits[:, :seq_len] = row.cuda()
    lengths = torch.full((num_rows,), seq_len, dtype=torch.int32, device="cuda")
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")

    torch.ops._C.cooperative_topk(logits, lengths, indices, top_k)

    ref = logits.float()[:, :seq_len].topk(top_k, dim=1).values
    got = logits.float().gather(1, indices.long()).sort(dim=1).values
    torch.testing.assert_close(got, ref)


def test_takes_bf16_logits_alignment() -> None:
    from sglang.kernels.jit.csrc.dsa.kpool_topk_transform import (  # type: ignore
        KpoolTopKTransformKernel,
    )

    aligned = torch.empty(4, 64, dtype=torch.bfloat16, device="cuda")
    unaligned = torch.empty(4, 65, dtype=torch.bfloat16, device="cuda")[:, :64]
    assert KpoolTopKTransformKernel.takes_bf16_logits(aligned)
    assert not KpoolTopKTransformKernel.takes_bf16_logits(unaligned)
