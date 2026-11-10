# Copyright (c) 2026, SGLang Team.
"""Routing checks for the FA4 SM90 d512 FP8-MMA path.

Runs on CPU. The policy module is imported by file path so this test does not
require the sglang package or a compiled FA4 extension; when torch is missing
the tensor-level cases are skipped explicitly instead of failing at import.
"""

import importlib.util
import pathlib

import pytest

_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")


def _load_policy():
    spec = importlib.util.spec_from_file_location(
        "_fa4_sm90_fp8_policy_under_test", _POLICY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = _load_policy()


def test_policy_module_is_torch_free_at_import():
    assert _POLICY_PATH.is_file()
    assert hasattr(policy, "is_sm90_fp8_kv_mma")
    assert hasattr(policy, "normalize_kv_descale")
    assert hasattr(policy, "select_num_splits")


def test_select_num_splits_fa3_keeps_cap():
    assert (
        policy.select_num_splits(
            fa_version=3,
            max_seqlen_k=8192,
            num_heads=32,
            num_kv_heads=8,
            num_sms=132,
            is_fp8_mma=False,
        )
        == 32
    )


def test_select_num_splits_non_fp8_passthrough():
    assert (
        policy.select_num_splits(
            fa_version=4,
            max_seqlen_k=8192,
            num_heads=32,
            num_kv_heads=8,
            num_sms=132,
            is_fp8_mma=False,
        )
        == 0
    )


@pytest.mark.parametrize(
    "max_seqlen_k,expected",
    [
        (0, 1),
        (64, 1),
        (256, 1),
        (1024, 4),
        (8192, 4),
    ],
)
def test_select_num_splits_fp8_bounded(max_seqlen_k, expected):
    assert (
        policy.select_num_splits(
            fa_version=4,
            max_seqlen_k=max_seqlen_k,
            num_heads=32,
            num_kv_heads=8,
            num_sms=132,
            is_fp8_mma=True,
        )
        == expected
    )


def test_select_num_splits_fp8_limited_by_sms():
    assert (
        policy.select_num_splits(
            fa_version=4,
            max_seqlen_k=8192,
            num_heads=32,
            num_kv_heads=8,
            num_sms=8,
            is_fp8_mma=True,
        )
        == 1
    )


def test_normalize_kv_descale_none_and_scalar():
    assert policy.normalize_kv_descale(None, batch=2, heads=4) is None
    assert policy.normalize_kv_descale(0.5, batch=2, heads=4) == 0.5


torch = pytest.importorskip("torch", reason="tensor descale cases need torch")


def test_normalize_kv_descale_layouts():
    batch, heads = 2, 4
    scalar = torch.tensor(0.25)
    out = policy.normalize_kv_descale(scalar, batch=batch, heads=heads)
    assert out.shape == (batch, heads)
    assert out.dtype == torch.float32
    assert out.is_contiguous()

    per_head = torch.arange(heads, dtype=torch.float32)
    out = policy.normalize_kv_descale(per_head, batch=batch, heads=heads)
    assert out.shape == (batch, heads)
    assert torch.equal(out[0], per_head)
    assert torch.equal(out[1], per_head)

    full = torch.arange(batch * heads, dtype=torch.float32).reshape(batch, heads)
    out = policy.normalize_kv_descale(full, batch=batch, heads=heads)
    assert torch.equal(out, full)

    four_d = full.reshape(batch, heads, 1, 1)
    out = policy.normalize_kv_descale(four_d, batch=batch, heads=heads)
    assert torch.equal(out, full)

    with pytest.raises(ValueError):
        policy.normalize_kv_descale(torch.ones(3), batch=batch, heads=heads)


def test_is_sm90_fp8_kv_mma_gating():
    q = torch.empty(1, 1, 8, 512, dtype=torch.bfloat16)
    k = torch.empty(1, 1, 1, 512, dtype=torch.float8_e4m3fn)
    v = torch.empty(1, 1, 1, 512, dtype=torch.float8_e4m3fn)
    assert policy.is_sm90_fp8_kv_mma(
        q, k, v, True, (None, None), device_capability_major=9
    )
    assert not policy.is_sm90_fp8_kv_mma(
        q, k, v, True, (None, None), device_capability_major=10
    )
    q256 = torch.empty(1, 1, 8, 256, dtype=torch.bfloat16)
    k256 = torch.empty(1, 1, 1, 256, dtype=torch.float8_e4m3fn)
    v256 = torch.empty(1, 1, 1, 256, dtype=torch.float8_e4m3fn)
    assert not policy.is_sm90_fp8_kv_mma(
        q256, k256, v256, True, (None, None), device_capability_major=9
    )
    with pytest.raises(NotImplementedError):
        policy.is_sm90_fp8_kv_mma(
            q, k, v, False, (None, None), device_capability_major=9
        )
