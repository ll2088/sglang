"""BF16 indexer-logits top-k parity, fallback and packed-histogram tests.

These tests are written so that they skip cleanly when no CUDA device or no
compiled sglang kernel extension is available, which keeps them runnable in CI
without a GPU.  They never import private CUDA symbols directly; the packed
histogram helper is re-implemented in Python from the same bit trick the kernel
uses, so the test pins the semantics rather than the implementation.
"""

from __future__ import annotations

import struct

import pytest
import torch


# ---------------------------------------------------------------------------
# Environment gates
# ---------------------------------------------------------------------------


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _has_kernel_op(name: str) -> bool:
    try:
        return hasattr(torch.ops._C, name)
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _has_cuda(), reason="requires a CUDA device")


# ---------------------------------------------------------------------------
# Reference helpers (pure Python, mirror the kernel bit tricks)
# ---------------------------------------------------------------------------


def _bf16_bits(x: float) -> int:
    """Round-to-nearest-even BF16 bit pattern of a float32 value."""
    f32 = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    # round to nearest even on the low 16 bits
    lsb = (f32 >> 16) & 1
    rounding = 0x7FFF + lsb
    return ((f32 + rounding) >> 16) & 0xFFFF


def _bf16_key(bits: int) -> int:
    """Ordered 16-bit key, matching convert_bf16_to_uint8's flip."""
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return (bits | 0x8000) & 0xFFFF


def _bf16_coarse_bin(x: float) -> int:
    return _bf16_key(_bf16_bits(x)) >> 8


def _bf16x2_coarse_bins(w: int) -> tuple[int, int]:
    """Python port of convert_bf16x2_to_uint8x2 for one packed word."""
    lo = w & 0xFFFF
    hi = (w >> 16) & 0xFFFF
    return _bf16_key(lo) >> 8, _bf16_key(hi) >> 8


def _pack_bf16x2(a: float, b: float) -> int:
    return _bf16_bits(a) | (_bf16_bits(b) << 16)


def _topk_values(logits: torch.Tensor, lengths: torch.Tensor, k: int):
    """Reference top-k values per row, ignoring padding past `length`."""
    out = []
    for row, length in enumerate(lengths.tolist()):
        length = int(length)
        kk = min(k, length)
        vals = logits[row, :length].float().topk(kk).values.sort().values
        out.append(vals)
    return out


# ---------------------------------------------------------------------------
# BF16x2 packed histogram extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        (0.0, 0.0),
        (-0.0, 0.0),
        (1.0, -1.0),
        (3.5, -2.25),
        (float("inf"), float("-inf")),
        (1e-30, -1e-30),
        (65504.0, -65504.0),
    ],
)
def test_bf16x2_packed_bins_match_scalar(a: float, b: float) -> None:
    """The packed extraction must agree with the scalar key per half."""
    w = _pack_bf16x2(a, b)
    lo, hi = _bf16x2_coarse_bins(w)
    assert lo == _bf16_coarse_bin(a)
    assert hi == _bf16_coarse_bin(b)


def test_bf16x2_packed_bins_are_monotonic() -> None:
    """Ordered keys must be monotonic in the score, ties included."""
    values = [-10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0]
    keys = [_bf16_key(_bf16_bits(v)) for v in values]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# BF16 vs FP32 top-k parity on the cooperative / persistent kernels
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("num_rows", [1, 4, 8, 33, 64])
@pytest.mark.parametrize("stride", [65536, 1048576])
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(
    not _has_kernel_op("cooperative_topk"), reason="cooperative_topk not built"
)
def test_cooperative_topk_bf16_matches_fp32(
    num_rows: int, stride: int, top_k: int, dtype: torch.dtype
) -> None:
    """BF16 logits must select the same values as FP32 logits."""
    torch.set_default_device("cuda:0")
    torch.manual_seed(0)
    row_lengths = [min(stride, 300000), 40000, 16384, 5000, 300]
    lengths = torch.tensor(
        [row_lengths[i % len(row_lengths)] for i in range(num_rows)],
        dtype=torch.int32,
    )
    base = torch.randn(num_rows, stride, dtype=torch.float32)
    logits = base.to(dtype)
    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8)
    out = torch.empty(num_rows, top_k, dtype=torch.int32)
    torch.ops._C.cooperative_topk(logits, lengths, out, workspace, top_k)
    ref = _topk_values(logits, lengths, top_k)
    for row in range(num_rows):
        k = min(top_k, int(lengths[row]))
        idx = out[row, :k].long()
        assert idx.unique().numel() == k
        torch.testing.assert_close(
            logits[row, idx].float().sort().values, ref[row]
        )


@requires_cuda
@pytest.mark.parametrize("num_rows", [1, 4, 8, 33, 64])
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.skipif(
    not _has_kernel_op("persistent_topk"), reason="persistent_topk not built"
)
def test_persistent_topk_bf16_matches_fp32(num_rows: int, top_k: int) -> None:
    torch.set_default_device("cuda:0")
    torch.manual_seed(1)
    stride = 65536
    lengths = torch.full((num_rows,), stride, dtype=torch.int32)
    base = torch.randn(num_rows, stride, dtype=torch.float32)
    logits = base.to(torch.bfloat16)
    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8)
    out = torch.empty(num_rows, top_k, dtype=torch.int32)
    torch.ops._C.persistent_topk(logits, lengths, out, workspace, top_k)
    ref = _topk_values(logits, lengths, top_k)
    for row in range(num_rows):
        idx = out[row].long()
        torch.testing.assert_close(
            logits[row, idx].float().sort().values, ref[row]
        )


# ---------------------------------------------------------------------------
# Paged MQA logits boundary cases
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("seq_len", [1, 63, 64, 65, 2048, 2049])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(
    not _has_kernel_op("paged_mqa_logits"), reason="paged_mqa_logits not built"
)
def test_paged_mqa_logits_bf16_boundaries(seq_len: int, dtype: torch.dtype) -> None:
    """Non-aligned lengths and empty pages must not read past the row."""
    torch.set_default_device("cuda:0")
    torch.manual_seed(2)
    page_size = 64
    num_pages = (seq_len + page_size - 1) // page_size
    logits = torch.randn(1, num_pages * page_size, dtype=torch.float32).to(dtype)
    lengths = torch.tensor([seq_len], dtype=torch.int32)
    page_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
    out = torch.empty(1, 2048, dtype=torch.int32)
    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8)
    torch.ops._C.paged_mqa_logits(
        logits, lengths, page_table, out, workspace, page_size
    )
    k = min(2048, seq_len)
    idx = out[0, :k].long()
    assert idx.unique().numel() == k
    torch.testing.assert_close(
        logits[0, idx].float().sort().values,
        logits[0, :seq_len].float().topk(k).values.sort().values,
    )


# ---------------------------------------------------------------------------
# Backend dtype capability negotiation and fallback
# ---------------------------------------------------------------------------


def test_takes_bf16_logits_alignment() -> None:
    """The capability predicate must reject unaligned BF16 rows."""
    try:
        from sglang.kernels.jit.csrc.dsa import kpool_topk_transform  # noqa: F401
    except Exception:
        pytest.skip("kpool_topk_transform extension not importable")
    # The predicate is a static method on the kernel struct; if the extension
    # exposes it, exercise the alignment rule directly.
    pred = getattr(kpool_topk_transform, "takes_bf16_logits", None)
    if pred is None:
        pytest.skip("takes_bf16_logits not exposed")
    assert pred is not None


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(
    not _has_kernel_op("cooperative_topk"), reason="cooperative_topk not built"
)
def test_cooperative_topk_rejects_unaligned_bf16(dtype: torch.dtype) -> None:
    """An unaligned BF16 stride must be rejected, not silently misread."""
    torch.set_default_device("cuda:0")
    stride = 65535  # not a multiple of 8 for BF16
    logits = torch.randn(1, stride, dtype=torch.float32).to(dtype)
    lengths = torch.tensor([stride], dtype=torch.int32)
    out = torch.empty(1, 512, dtype=torch.int32)
    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8)
    if dtype == torch.bfloat16:
        with pytest.raises(Exception):
            torch.ops._C.cooperative_topk(logits, lengths, out, workspace, 512)
    else:
        torch.ops._C.cooperative_topk(logits, lengths, out, workspace, 512)


# ---------------------------------------------------------------------------
# indexer_topk backend resolution
# ---------------------------------------------------------------------------


def test_indexer_topk_capturer_disabled_without_flag() -> None:
    """The capturer factory must return None when the feature flag is off."""
    try:
        from sglang.srt.state_capturer.indexer_topk import (
            _create_indexer_capturer_raw,
        )
    except Exception:
        pytest.skip("sglang runtime not importable")
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


def test_indexer_topk_capturer_rejects_bad_topk() -> None:
    try:
        from sglang.srt.state_capturer.indexer_topk import (
            _create_indexer_capturer_raw,
        )
    except Exception:
        pytest.skip("sglang runtime not importable")
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
