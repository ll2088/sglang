"""Standalone tests for the indexer_topk backend-resolution helpers.

These tests avoid importing torch/CUDA by exercising the pure-Python helpers
and by monkeypatching the runtime context lookups.
"""

from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path

import pytest


# Load the pure-Python helpers by explicit file path so importing them never
# triggers `sglang/__init__.py` (and therefore the full runtime dependency
# resolution).  The modules are kept free of torch/CUDA imports for exactly
# this reason.
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


def _load_indexer_topk_helpers():
    # Loaded lazily so collection never depends on the sglang runtime package.
    return _load_module(
        "_dsa_indexer_topk_helpers",
        "python/sglang/srt/state_capturer/indexer_topk.py",
    )


def test_extract_indexer_topk_from_meta_info_roundtrip():
    it = _load_indexer_topk_helpers()
    values = [1, 2, 3, 4]
    encoded = base64.b64encode(bytes(v & 0xFF for v in values)).decode("utf-8")
    data = {"meta_info": {"indexer_topk": encoded}}
    out = it.extract_indexer_topk_from_meta_info(data)
    assert list(out) == values


def test_extract_indexer_topk_missing_returns_empty():
    it = _load_indexer_topk_helpers()
    out = it.extract_indexer_topk_from_meta_info({"meta_info": {}})
    assert len(out) == 0


def test_maybe_capture_passthrough_without_capturer(monkeypatch):
    it = _load_indexer_topk_helpers()
    monkeypatch.setattr(it, "get_global_indexer_capturer", lambda: None)
    sentinel = object()
    assert it.maybe_capture_indexer_topk(0, sentinel) is sentinel
    assert it.maybe_capture_indexer_topk(0, None) is None


def test_maybe_capture_calls_capturer(monkeypatch):
    it = _load_indexer_topk_helpers()
    calls = []

    class _Cap:
        def capture(self, layer_id, topk_indices):
            calls.append((layer_id, topk_indices))

    monkeypatch.setattr(it, "get_global_indexer_capturer", lambda: _Cap())
    sentinel = object()
    assert it.maybe_capture_indexer_topk(3, sentinel) is sentinel
    assert calls == [(3, sentinel)]


def test_create_indexer_capturer_disabled_returns_none(monkeypatch):
    it = _load_indexer_topk_helpers()

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


def test_create_indexer_capturer_invalid_topk_returns_none(monkeypatch):
    it = _load_indexer_topk_helpers()
    out = it._create_indexer_capturer_raw(
        enable=True,
        num_indexer_layers=4,
        index_topk=0,
        num_tokens=16,
        max_running_requests=8,
        device="cuda",
    )
    assert out is None


def test_create_indexer_capturer_no_layers_returns_none():
    it = _load_indexer_topk_helpers()
    out = it._create_indexer_capturer_raw(
        enable=True,
        num_indexer_layers=0,
        index_topk=2048,
        num_tokens=16,
        max_running_requests=8,
        device="cuda",
    )
    assert out is None


def test_vec_width_is_derived_from_element_width():
    assert cap.vec_width(4) == 4
    assert cap.vec_width(2) == 8
    with pytest.raises(ValueError):
        cap.vec_width(3)


def test_takes_bf16_logits_requires_16_byte_alignment():
    assert cap.takes_bf16_logits(cap.BF16, 65536)
    assert cap.takes_bf16_logits(cap.BF16, 8)
    assert not cap.takes_bf16_logits(cap.BF16, 4)
    assert not cap.takes_bf16_logits(cap.BF16, 7)


def test_takes_bf16_logits_rejects_other_dtypes():
    assert not cap.takes_bf16_logits(cap.FP32, 65536)
    assert not cap.takes_bf16_logits(cap.DType(cap.DL_BFLOAT, 16, 2), 65536)
    assert not cap.takes_bf16_logits(cap.DType(cap.DL_BFLOAT, 32, 1), 65536)


def test_takes_fp32_logits_always_true_for_fp32():
    assert cap.takes_fp32_logits(cap.FP32)
    assert not cap.takes_fp32_logits(cap.BF16)


def test_resolve_logits_dtype_negotiates_and_falls_back():
    assert cap.resolve_logits_dtype(cap.BF16, 65536) == "bf16"
    # Misaligned BF16 rows fall back to the FP32 kernel after upcast.
    assert cap.resolve_logits_dtype(cap.BF16, 4) == "fp32"
    assert cap.resolve_logits_dtype(cap.FP32, 65536) == "fp32"
    with pytest.raises(ValueError):
        cap.resolve_logits_dtype(cap.DType(cap.DL_FLOAT, 16, 1), 65536)


def test_bf16x2_packed_extraction_matches_scalar():
    # +1.0, -1.0, +0.0, -0.0 as BF16 bit patterns.
    cases = [
        (0x3F80, 0xBF80),
        (0x0000, 0x8000),
        (0x7F7F, 0xFF7F),
        (0x3F80, 0x3F80),
    ]
    for lo, hi in cases:
        word = cap.pack_bf16x2(lo, hi)
        got = cap.host_convert_bf16x2_to_uint8x2(word)
        want = (cap.host_convert_bf16_to_uint8(lo), cap.host_convert_bf16_to_uint8(hi))
        assert got == want


def test_bf16x2_packed_extraction_is_order_preserving():
    # Ordered keys must be monotonic in the score for finite BF16 values.
    values = [0x0000, 0x3F80, 0x4000, 0x7F7F, 0x8000, 0xBF80, 0xC000, 0xFF7F]
    keys = [cap.host_convert_bf16_to_uint8(v) for v in values]
    assert keys == sorted(keys)


def test_bf16x2_packed_extraction_handles_high_half():
    word = cap.pack_bf16x2(0x3F80, 0xBF80)
    lo, hi = cap.host_convert_bf16x2_to_uint8x2(word)
    assert lo == cap.host_convert_bf16_to_uint8(0x3F80)
    assert hi == cap.host_convert_bf16_to_uint8(0xBF80)
