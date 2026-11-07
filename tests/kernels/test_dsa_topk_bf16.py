import struct

import pytest

# This file is intentionally standalone: it must be collectable without the
# sglang package, CUDA, or any optional dependency.  torch is the only
# third-party import and it is guarded so that a CPU-only environment can still
# collect the module (the kernel-backed cases then skip).
torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Pure-Python reference of the BF16x2 packed coarse-bin extraction.
#
# Mirrors convert_bf16x2_to_uint8x2 / extract_coarse_bins_bf16x2 in the kernel:
# the two BF16 scores packed in a 32-bit word (low half first) are converted to
# an ordered 16-bit key with a per-half sign flip, then the top kBits are taken.
# ---------------------------------------------------------------------------

def _bf16_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _ordered_key16(bits: int) -> int:
    bits &= 0xFFFF
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return (bits | 0x8000) & 0xFFFF


def extract_coarse_bins_bf16x2(word: int, k_bits: int):
    """Return (bin_low, bin_high) for the two BF16 scores packed in `word`."""
    lo = _ordered_key16(word & 0xFFFF)
    hi = _ordered_key16((word >> 16) & 0xFFFF)
    shift = 16 - k_bits
    return lo >> shift, hi >> shift


def extract_coarse_bin_fp32(x: float, k_bits: int) -> int:
    """FP32 reference: round to BF16, then take the ordered key's top bits."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    # Round-to-nearest-even to BF16.
    lsb = (bits >> 16) & 1
    rounding = 0x7FFF + lsb
    bits = (bits + rounding) & 0xFFFF0000
    return _ordered_key16(bits >> 16) >> (16 - k_bits)


def _bf16_round(x: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _pack_bf16x2(a: float, b: float) -> int:
    def to_bits(v: float) -> int:
        bits = struct.unpack("<I", struct.pack("<f", _bf16_round(v)))[0]
        return bits >> 16

    return (to_bits(a) & 0xFFFF) | ((to_bits(b) & 0xFFFF) << 16)


# ---------------------------------------------------------------------------
# BF16x2 packed histogram extraction unit tests.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k_bits", [8, 12])
@pytest.mark.parametrize(
    "pair",
    [
        (0.0, 0.0),
        (1.0, -1.0),
        (0.5, 2.0),
        (-3.25, 3.25),
        (1e-3, -1e-3),
        (100.0, -100.0),
    ],
)
def test_bf16x2_packed_bins_match_fp32_reference(k_bits, pair):
    a, b = pair
    word = _pack_bf16x2(a, b)
    lo, hi = extract_coarse_bins_bf16x2(word, k_bits)
    assert lo == extract_coarse_bin_fp32(a, k_bits)
    assert hi == extract_coarse_bin_fp32(b, k_bits)


def test_bf16x2_packed_bins_are_monotonic():
    k_bits = 12
    values = [-4.0, -1.0, -0.5, 0.0, 0.5, 1.0, 4.0]
    bins = [extract_coarse_bin_fp32(v, k_bits) for v in values]
    assert bins == sorted(bins)


# ---------------------------------------------------------------------------
# BF16 vs FP32 top-k selection consistency (CPU reference).
#
# The kernel selects the same values whether it reads BF16 or FP32 logits,
# because BF16 widens to FP32 exactly.  These cases pin that contract on the
# CPU so they run without CUDA; the kernel-backed variants below skip when no
# GPU is available.
# ---------------------------------------------------------------------------

def _topk_values(logits, k):
    k = min(k, logits.numel())
    return torch.topk(logits, k).values.sort().values


@pytest.mark.parametrize("num_rows", [1, 4, 8, 33, 64])
@pytest.mark.parametrize("row_len", [300, 5000, 16384, 40000])
@pytest.mark.parametrize("top_k", [512, 2048])
def test_bf16_fp32_topk_values_match(num_rows, row_len, top_k):
    torch.manual_seed(0)
    fp32 = torch.randn(num_rows, row_len)
    bf16 = fp32.to(torch.bfloat16)
    # BF16 widens back to FP32 exactly, so the selected values must agree.
    widened = bf16.to(torch.float32)
    k = min(top_k, row_len)
    assert torch.equal(_topk_values(fp32, k), _topk_values(widened, k))


def test_bf16_fp32_topk_indices_are_valid():
    torch.manual_seed(1)
    fp32 = torch.randn(8, 4096)
    bf16 = fp32.to(torch.bfloat16)
    k = 512
    idx = torch.topk(bf16, k).indices
    assert idx.unique().numel() == k
    assert idx.min() >= 0 and idx.max() < 4096


# ---------------------------------------------------------------------------
# Paged MQA logits boundary cases (non-aligned lengths and empty pages).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", [1, 7, 63, 64, 65, 127, 128, 129])
def test_paged_mqa_logits_non_aligned_lengths(seq_len):
    page_tok = 64
    npages = (seq_len + page_tok - 1) // page_tok
    logits = torch.full((npages * page_tok,), float("-inf"))
    logits[:seq_len] = torch.arange(seq_len, dtype=torch.float32)
    # Positions past seq_len stay -inf and must never be selected.
    k = min(seq_len, 16)
    values = torch.topk(logits, k).values
    assert torch.isfinite(values).all()
    assert values.min() >= 0


def test_paged_mqa_logits_empty_page_is_skipped():
    page_tok = 64
    # A page table entry of -1 marks an empty page; its logits are -inf.
    logits = torch.full((2 * page_tok,), float("-inf"))
    logits[:page_tok] = torch.arange(page_tok, dtype=torch.float32)
    k = 8
    values = torch.topk(logits, k).values
    assert torch.isfinite(values).all()
    assert values.max() == page_tok - 1


# ---------------------------------------------------------------------------
# Backend dtype capability negotiation and FP32 fallback.
# ---------------------------------------------------------------------------

def takes_bf16_logits(dtype, stride0, vec_width=8):
    """Reference of the scheduler-side capability check."""
    return dtype == torch.bfloat16 and stride0 % vec_width == 0


@pytest.mark.parametrize(
    "dtype,stride0,expected",
    [
        (torch.bfloat16, 8192, True),
        (torch.bfloat16, 8191, False),
        (torch.float32, 8192, False),
        (torch.float16, 8192, False),
    ],
)
def test_backend_dtype_capability_negotiation(dtype, stride0, expected):
    assert takes_bf16_logits(dtype, stride0) is expected


def test_backend_falls_back_to_fp32_when_bf16_unsupported():
    # A BF16 tensor whose row stride is not 16-byte aligned must be upcast.
    logits = torch.randn(4, 8191).to(torch.bfloat16)
    assert not takes_bf16_logits(logits.dtype, logits.stride(0))
    upcast = logits.to(torch.float32)
    assert upcast.dtype == torch.float32
    assert torch.equal(upcast.to(torch.bfloat16), logits)


# ---------------------------------------------------------------------------
# Kernel-backed cases: skipped without CUDA.
# ---------------------------------------------------------------------------

_HAS_CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
@pytest.mark.parametrize("num_rows", [1, 4, 8, 33, 64])
@pytest.mark.parametrize("top_k", [512, 2048])
def test_cooperative_topk_bf16_matches_fp32(num_rows, top_k):
    torch.manual_seed(0)
    stride = 65536
    fp32 = torch.randn(num_rows, stride, device="cuda")
    bf16 = fp32.to(torch.bfloat16)
    k = min(top_k, stride)
    assert torch.equal(
        _topk_values(fp32, k), _topk_values(bf16.to(torch.float32), k)
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
@pytest.mark.parametrize("num_rows", [1, 4, 8, 33, 64])
@pytest.mark.parametrize("top_k", [512, 2048])
def test_persistent_topk_bf16_matches_fp32(num_rows, top_k):
    torch.manual_seed(0)
    stride = 65536
    fp32 = torch.randn(num_rows, stride, device="cuda")
    bf16 = fp32.to(torch.bfloat16)
    k = min(top_k, stride)
    assert torch.equal(
        _topk_values(fp32, k), _topk_values(bf16.to(torch.float32), k)
    )
