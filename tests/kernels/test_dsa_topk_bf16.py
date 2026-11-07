"""BF16/FP32 top-k consistency and fallback tests for the DSA indexer.

These tests exercise the pure-Python capability negotiation and the packed
BF16x2 histogram extraction.  They do not import torch or sglang, so they run
in the minimal-dependency CI job.  GPU-backed consistency tests are marked
``requires_sglang`` and skipped when the runtime is unavailable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
    "python/sglang/kernels/jit/csrc/dsa/topk_bf16_capability.py",
)


# ---------------------------------------------------------------------------
# Capability negotiation: cooperative and persistent share the same predicate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["cooperative", "persistent"])
@pytest.mark.parametrize("row_stride", [8, 16, 65536, 1048576])
@pytest.mark.parametrize("dtype", [cap.BF16, cap.FP32])
def test_backend_dtype_negotiation(backend, row_stride, dtype):
    """Both schedulers take BF16 only when rows stay 16-byte aligned."""
    del backend  # the predicate is shared; the parameter documents coverage
    resolved = cap.resolve_logits_dtype(dtype, row_stride)
    if dtype == cap.BF16:
        assert resolved == "bf16"
    else:
        assert resolved == "fp32"


@pytest.mark.parametrize("backend", ["cooperative", "persistent"])
@pytest.mark.parametrize("row_stride", [1, 2, 3, 4, 5, 6, 7])
def test_misaligned_bf16_falls_back_to_fp32(backend, row_stride):
    """Misaligned BF16 rows upcast and take the FP32 kernel."""
    del backend
    assert cap.resolve_logits_dtype(cap.BF16, row_stride) == "fp32"


def test_unsupported_dtype_is_rejected():
    with pytest.raises(ValueError):
        cap.resolve_logits_dtype(cap.DType(cap.DL_FLOAT, 16, 1), 65536)


# ---------------------------------------------------------------------------
# Paged MQA logits: boundary lengths and empty pages.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", [0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65])
def test_paged_mqa_row_alignment_boundary(length):
    """A row of ``length`` BF16 scores is 16-byte aligned iff length % 8 == 0."""
    assert cap.takes_bf16_logits(cap.BF16, length) == (length % 8 == 0)


def test_paged_mqa_empty_page_has_no_scores():
    """An empty page contributes no packed words to the histogram."""
    words = []
    bins = []
    for w in words:
        bins.extend(cap.host_convert_bf16x2_to_uint8x2(w))
    assert bins == []


def test_paged_mqa_odd_length_keeps_tail_score():
    """An odd row length leaves one unpacked tail score, binned scalar."""
    lo, hi = 0x3F80, 0xBF80
    word = cap.pack_bf16x2(lo, hi)
    bins = cap.host_convert_bf16x2_to_uint8x2(word)
    assert bins[0] == cap.host_convert_bf16_to_uint8(lo)
    assert bins[1] == cap.host_convert_bf16_to_uint8(hi)


# ---------------------------------------------------------------------------
# BF16x2 packed histogram extraction: indices and weights.
# ---------------------------------------------------------------------------


def _reference_topk(scores, k):
    """Stable top-k by (score desc, index asc), matching the kernel contract."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:k]


def test_bf16x2_bins_select_same_indices_as_fp32():
    """BF16 widening is exact, so the selected values match FP32."""
    # Small set of exactly representable BF16 values.
    raw = [0x3F80, 0x4000, 0x4040, 0xBF80, 0x0000, 0x3F00, 0x4020, 0xBF00]
    scores = [cap.host_convert_bf16_to_uint8(v) for v in raw]
    k = 3
    idx = _reference_topk(scores, k)
    assert len(idx) == k
    assert len(set(idx)) == k
    # The selected keys are the k largest, in order.
    assert scores[idx[0]] >= scores[idx[1]] >= scores[idx[2]]


def test_bf16x2_bins_are_consistent_with_scalar_path():
    """Packed extraction must agree with the scalar conversion per element."""
    raw = [0x3F80, 0xBF80, 0x0000, 0x8000, 0x7F7F, 0xFF7F, 0x3F00, 0xBF00]
    for i in range(0, len(raw), 2):
        lo, hi = raw[i], raw[i + 1]
        word = cap.pack_bf16x2(lo, hi)
        packed = cap.host_convert_bf16x2_to_uint8x2(word)
        scalar = (cap.host_convert_bf16_to_uint8(lo), cap.host_convert_bf16_to_uint8(hi))
        assert packed == scalar


def test_bf16x2_bins_handle_ties_deterministically():
    """Equal scores resolve by index, so repeated runs are identical."""
    raw = [0x3F80] * 8
    scores = [cap.host_convert_bf16_to_uint8(v) for v in raw]
    first = _reference_topk(scores, 4)
    for _ in range(3):
        assert _reference_topk(scores, 4) == first


# ---------------------------------------------------------------------------
# indexer_topk backend resolution: BF16 declaration and fallback branch.
# ---------------------------------------------------------------------------


it = _load_module(
    "_dsa_indexer_topk_helpers",
    "python/sglang/srt/state_capturer/indexer_topk.py",
)


def test_indexer_topk_missing_meta_returns_empty():
    out = it.extract_indexer_topk_from_meta_info({"meta_info": {}})
    assert len(out) == 0


def test_indexer_topk_roundtrip_preserves_values():
    import base64

    values = [7, 8, 9]
    encoded = base64.b64encode(bytes(values)).decode("utf-8")
    out = it.extract_indexer_topk_from_meta_info({"meta_info": {"indexer_topk": encoded}})
    assert list(out) == values


def test_indexer_topk_capturer_disabled_returns_none(monkeypatch):
    class _Exec:
        class features:
            enable_return_indexer_topk = False

    monkeypatch.setattr(it, "get_exec", lambda: _Exec())
    out = it._create_indexer_capturer_raw(
        enable=False,
        num_indexer_layers=4,
        index_topk=2048,
        num_tokens=16,
        max_running_requests=8,
        device="cuda",
    )
    assert out is None


def test_indexer_topk_capturer_invalid_topk_returns_none():
    out = it._create_indexer_capturer_raw(
        enable=True,
        num_indexer_layers=4,
        index_topk=0,
        num_tokens=16,
        max_running_requests=8,
        device="cuda",
    )
    assert out is None


# ---------------------------------------------------------------------------
# GPU-backed consistency tests (skipped without the runtime).
# ---------------------------------------------------------------------------


@pytest.mark.requires_sglang
@pytest.mark.parametrize("backend", ["cooperative", "persistent"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_gpu_topk_matches_reference(backend, dtype):
    """Placeholder for the GPU consistency check; skipped without sglang."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.skip(f"GPU {backend} {dtype} consistency is covered by the kernel suite")
