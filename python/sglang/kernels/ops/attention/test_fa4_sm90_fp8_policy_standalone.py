# Copyright (c) 2026, SGLang Team.
"""Dependency-free routing tests for the FA4 SM90 d512 FP8-MMA policy.

This module imports only the standard library plus pytest, so CI can run it
without installing torch or the sglang package. It exercises the same pure
policy helpers as ``test_flash_attention_v4_fp8_routing.py`` but drives the
routing decision through explicit ``device_capability_major`` overrides and
lightweight tensor stand-ins instead of real CUDA tensors.
"""

import importlib.util
import pathlib
import sys

import pytest

_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")
_POLICY_MODULE_NAME = "_fa4_sm90_fp8_policy_standalone"
_spec = importlib.util.spec_from_file_location(_POLICY_MODULE_NAME, _POLICY_PATH)
fa4 = importlib.util.module_from_spec(_spec)
sys.modules[_POLICY_MODULE_NAME] = fa4
try:
    _spec.loader.exec_module(fa4)
except Exception:  # pragma: no cover - keep sys.modules clean on failure
    sys.modules.pop(_POLICY_MODULE_NAME, None)
    raise


class _FakeTensor:
    """Minimal tensor stand-in exposing only dtype/shape/ndim."""

    def __init__(self, dtype_name, head_dim=512, heads=8, batch=1):
        self.dtype = dtype_name
        self.shape = (batch, heads, head_dim)
        self.ndim = len(self.shape)


class _FakeDType:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, _FakeDType) and other.name == self.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"_FakeDType({self.name!r})"


def _tensor(dtype_name, head_dim=512, heads=8):
    return _FakeTensor(dtype_name, head_dim=head_dim, heads=heads)


def _call(q, k, v, *, causal=True, window_size=(None, None), major=9):
    return fa4.is_sm90_fp8_kv_mma(
        q, k, v, causal, window_size, device_capability_major=major
    )


def test_policy_module_has_no_heavy_imports():
    assert "torch" not in getattr(fa4, "__dict__", {})
    assert "sglang" not in getattr(fa4, "__dict__", {})
    assert fa4.SM90_FP8_MMA_HEAD_DIM == 512


@pytest.mark.parametrize("q_dtype", ["float16", "bfloat16"])
def test_sm90_d512_fp8_kv_routes_to_mma(q_dtype):
    q = _tensor(q_dtype)
    k = _tensor("float8_e4m3fn")
    v = _tensor("float8_e4m3fn")
    assert _call(q, k, v) is True


def test_non_sm90_falls_back():
    q = _tensor("bfloat16")
    k = _tensor("float8_e4m3fn")
    v = _tensor("float8_e4m3fn")
    assert _call(q, k, v, major=10) is False


def test_sm120_policy_unchanged():
    q = _tensor("bfloat16")
    k = _tensor("float8_e4m3fn")
    v = _tensor("float8_e4m3fn")
    assert _call(q, k, v, major=12) is False


def test_non_d512_falls_back():
    q = _tensor("bfloat16", head_dim=256)
    k = _tensor("float8_e4m3fn", head_dim=256)
    v = _tensor("float8_e4m3fn", head_dim=256)
    assert _call(q, k, v) is False


def test_non_fp8_kv_falls_back():
    q = _tensor("bfloat16")
    k = _tensor("bfloat16")
    v = _tensor("bfloat16")
    assert _call(q, k, v) is False


def test_sm90_fp8_kv_mma_requires_e4m3_kv():
    q = _tensor("bfloat16")
    k = _tensor("float8_e5m2")
    v = _tensor("float8_e5m2")
    assert _call(q, k, v) is False


def test_sm90_fp8_kv_mma_requires_equal_head_dims():
    q = _tensor("bfloat16", head_dim=512)
    k = _tensor("float8_e4m3fn", head_dim=512)
    v = _tensor("float8_e4m3fn", head_dim=256)
    assert _call(q, k, v) is False


def test_noncausal_full_attention_rejected():
    q = _tensor("bfloat16")
    k = _tensor("float8_e4m3fn")
    v = _tensor("float8_e4m3fn")
    with pytest.raises(NotImplementedError):
        _call(q, k, v, causal=False)


def test_noncausal_sliding_window_allowed():
    q = _tensor("bfloat16")
    k = _tensor("float8_e4m3fn")
    v = _tensor("float8_e4m3fn")
    assert _call(q, k, v, causal=False, window_size=(128, 0)) is True


def test_gemma4_full_attention_d512_shape_routes_to_fp8():
    q = _tensor("bfloat16", head_dim=512, heads=32)
    k = _tensor("float8_e4m3fn", head_dim=512, heads=8)
    v = _tensor("float8_e4m3fn", head_dim=512, heads=8)
    assert _call(q, k, v) is True


def test_gemma4_sliding_window_d256_stays_on_fa3():
    q = _tensor("bfloat16", head_dim=256, heads=32)
    k = _tensor("float8_e4m3fn", head_dim=256, heads=8)
    v = _tensor("float8_e4m3fn", head_dim=256, heads=8)
    assert _call(q, k, v, window_size=(128, 0)) is False


def test_stability_repeated_routing():
    q = _tensor("bfloat16")
    k = _tensor("float8_e4m3fn")
    v = _tensor("float8_e4m3fn")
    assert {_call(q, k, v) for _ in range(3)} == {True}


@pytest.mark.parametrize(
    "fa_version,is_fp8_mma,expected",
    [
        (3, False, 32),
        (3, True, 32),
        (4, False, 0),
        (4, True, 4),
    ],
)
def test_split_auto_selection(fa_version, is_fp8_mma, expected):
    splits = fa4.select_num_splits(
        fa_version=fa_version,
        max_seqlen_k=4096,
        num_heads=8,
        num_kv_heads=1,
        num_sms=32,
        is_fp8_mma=is_fp8_mma,
    )
    assert splits == expected


def test_split_auto_selection_short_seqlen():
    assert (
        fa4.select_num_splits(
            fa_version=4,
            max_seqlen_k=64,
            num_heads=8,
            num_kv_heads=1,
            num_sms=32,
            is_fp8_mma=True,
        )
        == 1
    )


def test_split_auto_selection_zero_seqlen():
    assert (
        fa4.select_num_splits(
            fa_version=4,
            max_seqlen_k=0,
            num_heads=8,
            num_kv_heads=1,
            num_sms=32,
            is_fp8_mma=True,
        )
        == 1
    )


def test_split_auto_selection_zero_sms():
    assert (
        fa4.select_num_splits(
            fa_version=4,
            max_seqlen_k=4096,
            num_heads=8,
            num_kv_heads=1,
            num_sms=0,
            is_fp8_mma=True,
        )
        == 1
    )


def test_split_auto_selection_zero_heads():
    assert (
        fa4.select_num_splits(
            fa_version=4,
            max_seqlen_k=4096,
            num_heads=0,
            num_kv_heads=1,
            num_sms=32,
            is_fp8_mma=True,
        )
        == 1
    )


def test_split_auto_selection_never_exceeds_tiles():
    assert (
        fa4.select_num_splits(
            fa_version=4,
            max_seqlen_k=256,
            num_heads=8,
            num_kv_heads=1,
            num_sms=1024,
            is_fp8_mma=True,
        )
        == 2
    )


def test_descale_none_passthrough():
    assert fa4.normalize_kv_descale(None, batch=2, heads=4) is None


def test_descale_layout_accepts_python_scalar():
    out = fa4.normalize_kv_descale(0.25, batch=2, heads=4)
    assert out.shape == (2, 4)
    assert out.dtype == "float32"


def test_descale_layout_rejects_bad_shape():
    with pytest.raises(ValueError):
        fa4.normalize_kv_descale(_FakeTensor("float32", head_dim=5, heads=2), batch=2, heads=4)
