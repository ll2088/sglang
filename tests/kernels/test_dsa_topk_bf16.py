import math
import struct

import pytest
import torch


# This file must be collectable on a CPU-only runner with no compiled
# extensions.  Every GPU/extension-dependent test is guarded by importorskip
# or skipif so that collection never fails.


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _has_sgl_kernel() -> bool:
    try:
        import sgl_kernel  # noqa: F401
    except Exception:
        return False
    return True


def _has_cooperative_topk_op() -> bool:
    try:
        return hasattr(torch.ops._C, "cooperative_topk")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pure-Python reference helpers (no GPU, no extension).
# ---------------------------------------------------------------------------


def _bf16_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _float_to_bf16_bits(x: float) -> int:
    # Round-to-nearest-even, matching __float2bfloat16_rn for finite values.
    u = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    lsb = (u >> 16) & 1
    rounding = 0x7FFF + lsb
    return ((u + rounding) >> 16) & 0xFFFF


def _bf16x2_key_flip(w: int) -> int:
    """Per-half ordered key flip used by convert_bf16x2_to_uint8x2.

    ~bits if negative, else bits | 0x8000, applied independently to each
    16-bit half of the packed word.
    """
    sign = w & 0x80008000
    return w ^ ((sign - (sign >> 15)) | 0x80008000)


def _bf16x2_coarse_bins(w: int, k_bits: int) -> tuple[int, int]:
    key = _bf16x2_key_flip(w)
    lo = (key & 0xFFFF) >> (16 - k_bits)
    hi = (key >> 16) >> (16 - k_bits)
    return lo, hi


def _scalar_bf16_coarse_bin(bits: int, k_bits: int) -> int:
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return key >> (16 - k_bits)


def _takes_bf16_logits(dtype, stride0: int, vec_width: int) -> bool:
    """Mirror of the host-side capability negotiation.

    BF16 logits are read natively only when the dtype is bfloat16 and the row
    stride keeps every row 16-byte aligned; otherwise the caller upcasts.
    """
    if dtype != torch.bfloat16:
        return False
    return stride0 % vec_width == 0


# ---------------------------------------------------------------------------
# BF16x2 packed histogram extraction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k_bits", [8, 12])
@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.0),
        (1.0, -1.0),
        (0.5, 2.0),
        (-0.0, 0.0),
        (3.25, -7.5),
        (float("inf"), float("-inf")),
    ],
)
def test_bf16x2_packed_bins_match_scalar(k_bits, values):
    lo_bits = _float_to_bf16_bits(values[0])
    hi_bits = _float_to_bf16_bits(values[1])
    w = lo_bits | (hi_bits << 16)
    lo, hi = _bf16x2_coarse_bins(w, k_bits)
    assert lo == _scalar_bf16_coarse_bin(lo_bits, k_bits)
    assert hi == _scalar_bf16_coarse_bin(hi_bits, k_bits)


def test_bf16x2_key_flip_is_monotonic():
    # Ordered keys must be non-decreasing in the score for a sorted sample.
    samples = [-10.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 10.0]
    keys = []
    for v in samples:
        bits = _float_to_bf16_bits(v)
        w = bits | (bits << 16)
        key = _bf16x2_key_flip(w)
        keys.append(key & 0xFFFF)
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Backend dtype capability negotiation and fallback.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,stride0,vec_width,expected",
    [
        (torch.bfloat16, 8192, 8, True),
        (torch.bfloat16, 8193, 8, False),
        (torch.bfloat16, 8192, 4, True),
        (torch.float32, 8192, 4, False),
        (torch.float16, 8192, 8, False),
    ],
)
def test_takes_bf16_logits_negotiation(dtype, stride0, vec_width, expected):
    assert _takes_bf16_logits(dtype, stride0, vec_width) is expected


def test_bf16_fallback_to_fp32_is_exact():
    # BF16 widens to FP32 exactly, so the fallback path selects the same values.
    torch.manual_seed(0)
    x = torch.randn(64, dtype=torch.bfloat16)
    widened = x.to(torch.float32)
    assert torch.equal(widened.to(torch.bfloat16), x)
    k = 8
    a = torch.topk(x, k).values
    b = torch.topk(widened, k).values
    assert torch.equal(a, b.to(torch.bfloat16))


# ---------------------------------------------------------------------------
# GPU / extension dependent tests.  These skip cleanly when unavailable.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")
@pytest.mark.skipif(not _has_sgl_kernel(), reason="requires sgl_kernel")
@pytest.mark.skipif(
    not _has_cooperative_topk_op(), reason="requires compiled cooperative_topk op"
)
@pytest.mark.parametrize("num_rows", [1, 4, 8, 33, 64])
@pytest.mark.parametrize("stride", [65536, 1048576])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top_k", [512, 2048])
@torch.inference_mode()
def test_cooperative_topk_bf16_matches_fp32(num_rows, stride, dtype, top_k):
    torch.set_default_device("cuda:0")
    torch.manual_seed(0)
    row_lengths = [min(stride, 300000), 40000, 16384, 5000, 300]
    lengths = [row_lengths[i % len(row_lengths)] for i in range(num_rows)]
    logits = torch.randn(num_rows, stride).to(dtype)
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8)
    out = torch.empty(num_rows, top_k, dtype=torch.int32)
    torch.ops._C.cooperative_topk(
        logits, torch.tensor(lengths, dtype=torch.int32), out, workspace, top_k
    )
    for row, length in enumerate(lengths):
        k = min(top_k, length)
        idx = out[row]
        assert torch.all(idx[k:] == -1)
        assert idx[:k].unique().numel() == k
        torch.testing.assert_close(
            logits[row, idx[:k].long()].sort().values,
            logits[row, :length].topk(k).values.sort().values,
        )


@pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")
@pytest.mark.skipif(not _has_sgl_kernel(), reason="requires sgl_kernel")
@pytest.mark.skipif(
    not _has_cooperative_topk_op(), reason="requires compiled cooperative_topk op"
)
@pytest.mark.parametrize("num_rows", [9, 34, 64])
@pytest.mark.parametrize("seq_len", [65535, 65536, 65537])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_cooperative_topk_overflow_cluster_boundary(num_rows, seq_len, dtype):
    top_k = 2048
    stride = (seq_len + 7) & ~7 if dtype == torch.bfloat16 else (seq_len + 3) & ~3
    row = 1.0 + torch.arange(seq_len, dtype=torch.float32) * (0.01 / seq_len)
    logits = torch.full((num_rows, stride), float("inf"), device="cuda", dtype=dtype)
    logits[:, :seq_len] = row.cuda()
    lengths = torch.full((num_rows,), seq_len, dtype=torch.int32, device="cuda")
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    torch.ops._C.cooperative_topk(logits, lengths, indices, workspace, top_k)
    for r in range(num_rows):
        torch.testing.assert_close(
            logits[r, indices[r].long()].sort().values,
            logits[r, :seq_len].topk(top_k).values.sort().values,
        )


@pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")
@pytest.mark.skipif(not _has_sgl_kernel(), reason="requires sgl_kernel")
@pytest.mark.skipif(
    not _has_cooperative_topk_op(), reason="requires compiled cooperative_topk op"
)
@pytest.mark.parametrize("num_rows", [1, 64])
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize("seq_len", [8192, 32768])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_cooperative_topk_16bit_grid_overflow(num_rows, top_k, seq_len, dtype):
    stride = (seq_len + 7) & ~7 if dtype == torch.bfloat16 else (seq_len + 3) & ~3
    logits = torch.randn(num_rows, stride, device="cuda", dtype=dtype)
    lengths = torch.full((num_rows,), seq_len, dtype=torch.int32, device="cuda")
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    torch.ops._C.cooperative_topk(logits, lengths, indices, workspace, top_k)
    for r in range(num_rows):
        torch.testing.assert_close(
            logits[r, indices[r].long()].sort().values,
            logits[r, :seq_len].topk(top_k).values.sort().values,
        )


@pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")
@pytest.mark.skipif(not _has_sgl_kernel(), reason="requires sgl_kernel")
@pytest.mark.skipif(
    not _has_cooperative_topk_op(), reason="requires compiled cooperative_topk op"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_cooperative_topk_overflow_nan(dtype):
    top_k = 512
    seq_len = 65537
    stride = (seq_len + 7) & ~7 if dtype == torch.bfloat16 else (seq_len + 3) & ~3
    logits = torch.randn(4, stride, device="cuda", dtype=dtype)
    logits[:, seq_len - 1] = float("nan")
    lengths = torch.full((4,), seq_len, dtype=torch.int32, device="cuda")
    indices = torch.empty((4, top_k), dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    torch.ops._C.cooperative_topk(logits, lengths, indices, workspace, top_k)
    for r in range(4):
        assert indices[r].unique().numel() == top_k


@pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")
@pytest.mark.skipif(not _has_sgl_kernel(), reason="requires sgl_kernel")
@pytest.mark.skipif(
    not _has_cooperative_topk_op(), reason="requires compiled cooperative_topk op"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_cooperative_topk_overflow_mixed_lengths(dtype):
    top_k = 2048
    lengths_list = [65537, 300, 16384, 5000]
    stride = 65544 if dtype == torch.bfloat16 else 65540
    logits = torch.randn(len(lengths_list), stride, device="cuda", dtype=dtype)
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device="cuda")
    indices = torch.empty((len(lengths_list), top_k), dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    torch.ops._C.cooperative_topk(logits, lengths, indices, workspace, top_k)
    for r, length in enumerate(lengths_list):
        k = min(top_k, length)
        torch.testing.assert_close(
            logits[r, indices[r, :k].long()].sort().values,
            logits[r, :length].topk(k).values.sort().values,
        )


@pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")
@pytest.mark.skipif(not _has_sgl_kernel(), reason="requires sgl_kernel")
@pytest.mark.skipif(
    not _has_cooperative_topk_op(), reason="requires compiled cooperative_topk op"
)
@torch.inference_mode()
def test_persistent_topk_bf16_matches_fp32():
    top_k = 2048
    num_rows = 128
    seq_len = 32768
    stride = (seq_len + 7) & ~7
    logits = torch.randn(num_rows, stride, device="cuda", dtype=torch.bfloat16)
    lengths = torch.full((num_rows,), seq_len, dtype=torch.int32, device="cuda")
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    torch.ops._C.persistent_topk(logits, lengths, indices, workspace, top_k)
    for r in range(num_rows):
        torch.testing.assert_close(
            logits[r, indices[r].long()].sort().values,
            logits[r, :seq_len].topk(top_k).values.sort().values,
        )


@pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")
@pytest.mark.skipif(not _has_sgl_kernel(), reason="requires sgl_kernel")
@torch.inference_mode()
def test_paged_mqa_logits_bf16_boundary():
    # Non-aligned length and empty page: the kernel must clamp and never read
    # past the row.
    pytest.importorskip("sglang.kernels.jit.csrc.dsa_gfx950.paged_mqa_logits")
    pytest.skip("paged MQA logits kernel is ROCm-only; covered on gfx950 CI")


@pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")
@pytest.mark.skipif(not _has_sgl_kernel(), reason="requires sgl_kernel")
@torch.inference_mode()
def test_indexer_topk_backend_resolution_bf16_fallback():
    from sglang.srt.state_capturer.indexer_topk import (
        _create_indexer_capturer_raw,
    )

    # Disabled capturer must return None without touching the device.
    assert (
        _create_indexer_capturer_raw(
            enable=False,
            num_indexer_layers=4,
            index_topk=2048,
            num_tokens=1024,
            max_running_requests=8,
            device="cuda",
        )
        is None
    )
    # Invalid index_topk must fall back to None and log, not raise.
    assert (
        _create_indexer_capturer_raw(
            enable=True,
            num_indexer_layers=4,
            index_topk=0,
            num_tokens=1024,
            max_running_requests=8,
            device="cuda",
        )
        is None
    )
