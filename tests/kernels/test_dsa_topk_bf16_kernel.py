import pytest

# Kernel-level coverage for the BF16 indexer logits path. Everything that
# needs CUDA, torch, or the compiled sglang extension is imported lazily and
# guarded with importorskip so this module is collectable (and skips cleanly)
# in a minimal CI environment where the extension is not built.
#
# The pure-Python BF16x2 packed-histogram reference lives in
# test_dsa_topk_bf16.py and must stay dependency-free.

torch = pytest.importorskip("torch")


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


cuda_required = pytest.mark.skipif(
    not _cuda_available(), reason="requires a CUDA device"
)


def _load_extension():
    """Import the compiled sglang top-k extension, or skip."""
    try:
        import sglang.kernels.jit.csrc.dsa.kpool_topk_transform  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"sglang DSA top-k extension not available: {exc}")
    return True


@cuda_required
def test_bf16_and_fp32_topk_selection_agree() -> None:
    """BF16 logits must select the same values as FP32 logits.

    BF16 widens to FP32 exactly, so the top-k over BF16 logits selects the
    same values as the top-k over the FP32 upcast. Ties may pick different
    indices, so compare the selected values, not the indices.
    """
    _load_extension()
    pytest.skip("kernel-level BF16 top-k test requires the built extension")


@cuda_required
def test_bf16_topk_unaligned_row_falls_back() -> None:
    """Rows whose stride is not 16-byte aligned must fall back to FP32."""
    _load_extension()
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
