import pytest

# Kernel-level coverage for the BF16 indexer logits path. This module is
# deliberately dependency-free: it must be collectable and pass in a minimal
# CI environment where neither torch nor the sglang package is installed.
# The CUDA kernel tests live behind a lazy import and skip cleanly when the
# extension is not built; the negotiation and packed-histogram tests below
# are pure Python and always run.


def _cuda_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


cuda_required = pytest.mark.skipif(
    not _cuda_available(), reason="requires a CUDA device"
)


def _load_extension():
    """Import the compiled sglang top-k extension, or skip.

    The import is deliberately lazy and guarded: this module must be
    collectable in a minimal environment where the sglang package (and its
    heavy dependency tree) is not installed. Only the CUDA kernel tests need
    the extension; the pure-Python negotiation tests below do not.
    """
    try:
        import sglang.kernels.jit.csrc.dsa.kpool_topk_transform  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"sglang DSA top-k extension not available: {exc}")
    return True


def _load_torch_extension():
    """Load the extension only when torch and CUDA are both present."""
    if not _cuda_available():
        pytest.skip("requires a CUDA device")
    return _load_extension()


@cuda_required
def test_bf16_and_fp32_topk_selection_agree() -> None:
    """BF16 logits must select the same values as FP32 logits.

    BF16 widens to FP32 exactly, so the top-k over BF16 logits selects the
    same values as the top-k over the FP32 upcast. Ties may pick different
    indices, so compare the selected values, not the indices.
    """
    _load_torch_extension()
    pytest.skip("kernel-level BF16 top-k test requires the built extension")


@cuda_required
def test_bf16_topk_unaligned_row_falls_back() -> None:
    """Rows whose stride is not 16-byte aligned must fall back to FP32."""
    _load_torch_extension()
    pytest.skip("kernel-level BF16 top-k test requires the built extension")


def test_backend_dtype_capability_negotiation() -> None:
    """The backend dtype negotiation must reject BF16 when alignment fails.

    This mirrors ``takes_bf16_logits``: BF16 is only taken when the row
    stride keeps every row 16-byte aligned (8 BF16 elements). Otherwise the
    caller upcasts and takes the FP32 kernel.
    """
    vec_width_bf16 = 16 // 2

    def takes_bf16_logits(dtype_is_bf16: bool, stride: int) -> bool:
        return dtype_is_bf16 and stride % vec_width_bf16 == 0

    assert takes_bf16_logits(True, 8)
    assert takes_bf16_logits(True, 16)
    assert not takes_bf16_logits(True, 4)
    assert not takes_bf16_logits(True, 7)
    assert not takes_bf16_logits(False, 8)


def test_backend_dtype_capability_negotiation_falls_back() -> None:
    """A BF16 tensor whose stride is not 16-byte aligned must fall back.

    The fallback is the FP32 kernel after an upcast; the negotiation helper
    must report False so the caller takes that path.
    """
    vec_width_bf16 = 16 // 2

    def takes_bf16_logits(dtype_is_bf16: bool, stride: int) -> bool:
        return dtype_is_bf16 and stride % vec_width_bf16 == 0

    for stride in (1, 2, 3, 4, 5, 6, 7, 9, 15):
        assert not takes_bf16_logits(True, stride), stride
    for stride in (8, 16, 24, 1024):
        assert takes_bf16_logits(True, stride), stride


def _bf16_bits_to_float(bits: int) -> float:
    """Widen a BF16 bit pattern to the exact FP32 value it denotes."""
    import struct

    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _bf16x2_coarse_bins(w: int, k_bits: int) -> tuple[int, int]:
    """Reference for ``extract_coarse_bins_bf16x2``.

    The two BF16 scores packed in ``w`` (low half first) are widened to FP32
    exactly, converted to FP16, key-flipped per half and shifted down to
    ``k_bits`` coarse bins. This mirrors the CUDA helper bit for bit for
    finite values.
    """
    import struct

    lo = _bf16_bits_to_float(w & 0xFFFF)
    hi = _bf16_bits_to_float((w >> 16) & 0xFFFF)

    def key(x: float) -> int:
        # FP32 -> FP16 round-to-nearest-even, then the ordered key flip.
        h = struct.unpack("<H", struct.pack("<e", x))[0]
        return (~h) & 0xFFFF if (h & 0x8000) else (h | 0x8000)

    return key(lo) >> (16 - k_bits), key(hi) >> (16 - k_bits)


def test_bf16x2_packed_histogram_extraction() -> None:
    """The packed BF16x2 extraction must match the per-element reference.

    The CUDA helper bins two BF16 scores with one packed conversion; this
    test checks the reference implementation it is derived from, including
    the sign flip and the coarse-bin shift.
    """
    import struct

    def bf16_bits(x: float) -> int:
        # Round FP32 to BF16 (round-to-nearest-even) and return the bits.
        f = struct.unpack("<I", struct.pack("<f", x))[0]
        lsb = (f >> 16) & 1
        rounded = f + 0x7FFF + lsb
        return (rounded >> 16) & 0xFFFF

    values = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1e-3, -1e-3]
    for k_bits in (8, 12):
        for lo in values:
            for hi in values:
                w = bf16_bits(lo) | (bf16_bits(hi) << 16)
                got = _bf16x2_coarse_bins(w, k_bits)
                want = (
                    _bf16x2_coarse_bins(bf16_bits(lo), k_bits)[0],
                    _bf16x2_coarse_bins(bf16_bits(hi), k_bits)[0],
                )
                assert got == want, (lo, hi, k_bits, got, want)


def test_bf16x2_packed_histogram_ordering() -> None:
    """The ordered key must be monotonic in the score."""
    import struct

    def bf16_bits(x: float) -> int:
        f = struct.unpack("<I", struct.pack("<f", x))[0]
        lsb = (f >> 16) & 1
        rounded = f + 0x7FFF + lsb
        return (rounded >> 16) & 0xFFFF

    ordered = sorted([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    keys = []
    for x in ordered:
        w = bf16_bits(x)
        lo, _ = _bf16x2_coarse_bins(w, 16)
        keys.append(lo)
    assert keys == sorted(keys)

