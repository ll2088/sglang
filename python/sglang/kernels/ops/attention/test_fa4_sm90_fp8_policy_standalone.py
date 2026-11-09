# Copyright (c) 2026, SGLang Team.
"""Standalone routing tests for the FA4 SM90 d512 FP8-MMA policy.

Loads the pure policy module by path so this file runs with only pytest
installed (no torch, no sglang). Tensor-based cases are skipped when torch
is unavailable.
"""

import importlib.util
import pathlib
import sys

import pytest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="torch is required")

_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")
_NAME = "_fa4_sm90_fp8_policy_standalone"
_spec = importlib.util.spec_from_file_location(_NAME, _POLICY_PATH)
fa4 = importlib.util.module_from_spec(_spec)
sys.modules[_NAME] = fa4
_spec.loader.exec_module(fa4)


def _tensor(dtype, head_dim=512, heads=8):
    return torch.empty(1, heads, head_dim, dtype=dtype)


def test_policy_module_has_no_torch_or_sglang_at_import():
    assert "torch" not in fa4.__dict__
    assert "sglang" not in fa4.__dict__
    assert fa4.SM90_FP8_MMA_HEAD_DIM == 512


@requires_torch
@pytest.mark.parametrize("q_dtype", ["float16", "bfloat16"])
def test_sm90_d512_fp8_kv_routes_to_mma(q_dtype):
    q = _tensor(getattr(torch, q_dtype))
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    assert fa4.is_sm90_fp8_kv_mma(
        q, k, v, True, (None, None), device_capability_major=9
    ) is True


@requires_torch
@pytest.mark.parametrize("major", [8, 10, 12])
def test_non_sm90_falls_back(major):
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    assert fa4.is_sm90_fp8_kv_mma(
        q, k, v, True, (None, None), device_capability_major=major
    ) is False


@requires_torch
def test_non_d512_falls_back():
    q = _tensor(torch.bfloat16, head_dim=256)
    k = _tensor(torch.float8_e4m3fn, head_dim=256)
    v = _tensor(torch.float8_e4m3fn, head_dim=256)
    assert fa4.is_sm90_fp8_kv_mma(
        q, k, v, True, (None, None), device_capability_major=9
    ) is False


@requires_torch
def test_non_fp8_kv_falls_back():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.bfloat16)
    v = _tensor(torch.bfloat16)
    assert fa4.is_sm90_fp8_kv_mma(
        q, k, v, True, (None, None), device_capability_major=9
    ) is False


@requires_torch
def test_noncausal_full_attention_rejected():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    with pytest.raises(NotImplementedError):
        fa4.is_sm90_fp8_kv_mma(
            q, k, v, False, (None, None), device_capability_major=9
        )


@requires_torch
def test_noncausal_sliding_window_allowed():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    assert fa4.is_sm90_fp8_kv_mma(
        q, k, v, False, (128, 0), device_capability_major=9
    ) is True


@requires_torch
@pytest.mark.parametrize(
    "layout,expected_shape",
    [
        (0.5, (2, 4)),
        ("ones1", (2, 4)),
        ("ones4", (2, 4)),
        ("ones24", (2, 4)),
        ("ones2411", (2, 4)),
    ],
)
def test_descale_layout_normalization(layout, expected_shape):
    if isinstance(layout, str):
        layout = {
            "ones1": torch.ones(1),
            "ones4": torch.ones(4),
            "ones24": torch.ones(2, 4),
            "ones2411": torch.ones(2, 4, 1, 1),
        }[layout]
    out = fa4.normalize_kv_descale(layout, batch=2, heads=4)
    assert out.shape == expected_shape
    assert out.dtype == torch.float32
    assert out.is_contiguous()


@requires_torch
@pytest.mark.parametrize(
    "bad",
    ["ones3", "ones352", "ones241", "ones2421"],
)
def test_descale_layout_rejects_bad_shape(bad):
    t = {
        "ones3": torch.ones(3),
        "ones352": torch.ones(3, 5, 2),
        "ones241": torch.ones(2, 4, 1),
        "ones2421": torch.ones(2, 4, 2, 1),
    }[bad]
    with pytest.raises(ValueError):
        fa4.normalize_kv_descale(t, batch=2, heads=4)


def test_descale_none_passthrough():
    assert fa4.normalize_kv_descale(None, batch=2, heads=4) is None


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
    assert fa4.select_num_splits(
        fa_version=fa_version,
        max_seqlen_k=4096,
        num_heads=8,
        num_kv_heads=1,
        num_sms=32,
        is_fp8_mma=is_fp8_mma,
    ) == expected


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (dict(max_seqlen_k=64, num_sms=32), 1),
        (dict(max_seqlen_k=0, num_sms=32), 1),
        (dict(max_seqlen_k=4096, num_sms=0), 1),
        (dict(max_seqlen_k=4096, num_sms=8), 1),
        (dict(max_seqlen_k=256, num_sms=1024), 4),
    ],
)
def test_split_auto_selection_edges(kwargs, expected):
    assert fa4.select_num_splits(
        fa_version=4,
        num_heads=8,
        num_kv_heads=1,
        is_fp8_mma=True,
        **kwargs,
    ) == expected


def test_split_auto_selection_zero_heads():
    assert fa4.select_num_splits(
        fa_version=4,
        max_seqlen_k=4096,
        num_heads=0,
        num_kv_heads=1,
        num_sms=32,
        is_fp8_mma=True,
    ) == 1
