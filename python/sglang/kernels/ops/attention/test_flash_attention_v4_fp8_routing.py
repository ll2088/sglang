# Copyright (c) 2026, SGLang Team.
"""Routing tests for the FA4 SM90 d512 FP8-MMA path (no GPU required)."""

import importlib.util
import pathlib
import sys
from unittest.mock import patch

import pytest

# The pure policy module is loaded by path below and must not import torch or
# sglang at module scope, so the routing checks run on a CPU-only runner.
torch = pytest.importorskip("torch", reason="torch is required for routing tests")

# The policy module is loaded by path and must not import sglang or CUDA at
# module scope, so these routing checks run on a CPU-only runner.

# This module must not import sglang or initialize CUDA: CI installs sglang
# from PyPI and that step can fail before pytest runs. The policy module is
# loaded by path below and stays torch-free at module scope.

# The policy module is loaded by path below; it must never import torch at
# module scope so that this file stays runnable without the sglang package.

# This test intentionally avoids importing the sglang package: CI installs
# sglang from PyPI and that step can fail before pytest runs. Loading the
# pure policy module by path keeps the routing checks runnable standalone.

# Load the pure policy module by path so this test does not require the
# sglang package (or its heavy runtime deps) to be importable. The module is
# registered in sys.modules before exec so dataclasses/typing introspection
# inside the module cannot re-trigger a package import.
_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")
_POLICY_MODULE_NAME = "_fa4_sm90_fp8_policy_under_test"
_spec = importlib.util.spec_from_file_location(_POLICY_MODULE_NAME, _POLICY_PATH)
fa4 = importlib.util.module_from_spec(_spec)
sys.modules[_POLICY_MODULE_NAME] = fa4
try:
    _spec.loader.exec_module(fa4)
except Exception:  # pragma: no cover - keep sys.modules clean on failure
    sys.modules.pop(_POLICY_MODULE_NAME, None)
    raise

_normalize_descale = fa4.normalize_kv_descale
_select_num_splits = fa4.select_num_splits
_is_sm90_fp8_kv_mma = fa4.is_sm90_fp8_kv_mma


class _FakeCuda:
    def __init__(self, major: int):
        self.major = major

    def get_device_capability(self, device=None):
        return (self.major, 0)


def _tensor(dtype, head_dim=512, heads=8):
    return torch.empty(1, heads, head_dim, dtype=dtype)


def _call(q, k, v, *, causal=True, window_size=(None, None)):
    return fa4.is_sm90_fp8_kv_mma(q, k, v, causal, window_size)


@pytest.mark.parametrize("major", [9])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_sm90_d512_fp8_kv_routes_to_mma(major, q_dtype):
    q = _tensor(q_dtype)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    with patch.object(fa4.torch, "cuda", _FakeCuda(major)):
        assert _call(q, k, v) is True


def test_non_sm90_falls_back():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    with patch.object(fa4.torch, "cuda", _FakeCuda(10)):
        assert _call(q, k, v) is False


def test_non_d512_falls_back():
    q = _tensor(torch.bfloat16, head_dim=256)
    k = _tensor(torch.float8_e4m3fn, head_dim=256)
    v = _tensor(torch.float8_e4m3fn, head_dim=256)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        assert _call(q, k, v) is False


def test_non_fp8_kv_falls_back():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.bfloat16)
    v = _tensor(torch.bfloat16)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        assert _call(q, k, v) is False


def test_noncausal_full_attention_rejected():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        with pytest.raises(NotImplementedError):
            _call(q, k, v, causal=False)


def test_noncausal_sliding_window_allowed():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        assert _call(q, k, v, causal=False, window_size=(128, 0)) is True


@pytest.mark.parametrize(
    "layout,expected_shape",
    [
        (0.5, (2, 4)),
        (torch.ones(4), (2, 4)),
        (torch.ones(2, 4), (2, 4)),
        (torch.ones(2, 4, 1, 1), (2, 4)),
    ],
)
def test_descale_layout_normalization(layout, expected_shape):
    out = _normalize_descale(layout, batch=2, heads=4)
    assert out.shape == expected_shape
    assert out.dtype == torch.float32
    assert out.is_contiguous()


def test_descale_layout_rejects_bad_shape():
    with pytest.raises(ValueError):
        _normalize_descale(torch.ones(3, 5, 2), batch=2, heads=4)


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
    splits = _select_num_splits(
        fa_version=fa_version,
        max_seqlen_k=4096,
        num_heads=8,
        num_kv_heads=1,
        num_sms=32,
        is_fp8_mma=is_fp8_mma,
    )
    assert splits == expected


def test_split_auto_selection_short_seqlen():
    splits = _select_num_splits(
        fa_version=4,
        max_seqlen_k=64,
        num_heads=8,
        num_kv_heads=1,
        num_sms=32,
        is_fp8_mma=True,
    )
    assert splits == 1


def test_non_fp8_fallback_keeps_original_kernel():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.bfloat16)
    v = _tensor(torch.bfloat16)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        assert _call(q, k, v) is False


def test_non_sm90_d512_fp8_falls_back():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    with patch.object(fa4.torch, "cuda", _FakeCuda(10)):
        assert _call(q, k, v) is False


def test_policy_module_does_not_import_torch_at_module_scope():
    # Guards the CI fix: importing the policy module must not require torch.
    assert "torch" not in getattr(fa4, "__dict__", {})


def test_policy_module_does_not_import_sglang():
    # Guards the CI fix: the policy module must not pull in the sglang package.
    assert "sglang" not in getattr(fa4, "__dict__", {})


def test_non_fp8_fallback_keeps_original_kernel_sm90():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.bfloat16)
    v = _tensor(torch.bfloat16)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        assert _call(q, k, v) is False


def test_sm120_policy_unchanged():
    # SM120 keeps its own dispatch policy; the SM90 FP8-MMA gate must not fire.
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    with patch.object(fa4.torch, "cuda", _FakeCuda(12)):
        assert _call(q, k, v) is False


def test_policy_module_imports_without_torch():
    # Guards the CI fix: the policy module must import even when torch is
    # unavailable, so routing tests can run on a CPU-only runner.
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch blocked for test")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", _blocked):
        spec = importlib.util.spec_from_file_location(
            "_fa4_sm90_fp8_policy_no_torch", _POLICY_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    assert mod.SM90_FP8_MMA_HEAD_DIM == 512


def test_stability_repeated_routing():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        results = {_call(q, k, v) for _ in range(3)}
    assert results == {True}


def test_stability_repeated_descale_and_splits():
    descale = torch.ones(2, 4)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        for _ in range(3):
            out = _normalize_descale(descale, batch=2, heads=4)
            assert out.shape == (2, 4)
            assert out.dtype == torch.float32
            splits = _select_num_splits(
                fa_version=4,
                max_seqlen_k=4096,
                num_heads=8,
                num_kv_heads=1,
                num_sms=32,
                is_fp8_mma=True,
            )
            assert splits == 4


def test_descale_none_passthrough():
    assert _normalize_descale(None, batch=2, heads=4) is None


def test_split_auto_selection_zero_seqlen():
    splits = _select_num_splits(
        fa_version=4,
        max_seqlen_k=0,
        num_heads=8,
        num_kv_heads=1,
        num_sms=32,
        is_fp8_mma=True,
    )
    assert splits == 1


def test_split_auto_selection_zero_sms():
    splits = _select_num_splits(
        fa_version=4,
        max_seqlen_k=4096,
        num_heads=8,
        num_kv_heads=1,
        num_sms=0,
        is_fp8_mma=True,
    )
    assert splits == 1


def test_split_auto_selection_zero_heads():
    splits = _select_num_splits(
        fa_version=4,
        max_seqlen_k=4096,
        num_heads=0,
        num_kv_heads=1,
        num_sms=32,
        is_fp8_mma=True,
    )
    assert splits == 1


def test_split_auto_selection_never_exceeds_tiles():
    splits = _select_num_splits(
        fa_version=4,
        max_seqlen_k=256,
        num_heads=8,
        num_kv_heads=1,
        num_sms=1024,
        is_fp8_mma=True,
    )
    assert splits == 2


def test_descale_layout_rejects_bad_1d():
    with pytest.raises(ValueError):
        _normalize_descale(torch.ones(3), batch=2, heads=4)


def test_descale_layout_rejects_bad_4d_trailing():
    with pytest.raises(ValueError):
        _normalize_descale(torch.ones(2, 4, 2, 1), batch=2, heads=4)


def test_descale_layout_rejects_unsupported_ndim():
    with pytest.raises(ValueError):
        _normalize_descale(torch.ones(2, 4, 1), batch=2, heads=4)


def test_descale_layout_accepts_python_scalar():
    out = _normalize_descale(0.25, batch=2, heads=4)
    assert out.shape == (2, 4)
    assert out.dtype == torch.float32
    assert torch.all(out == 0.25)


def test_descale_layout_preserves_values():
    src = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    out = _normalize_descale(src, batch=2, heads=4)
    assert torch.equal(out, src)
    assert out.dtype == torch.float32


def test_sm90_fp8_kv_mma_requires_e4m3_kv():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e5m2)
    v = _tensor(torch.float8_e5m2)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        assert _call(q, k, v) is False


def test_sm90_fp8_kv_mma_requires_equal_head_dims():
    q = _tensor(torch.bfloat16, head_dim=512)
    k = _tensor(torch.float8_e4m3fn, head_dim=512)
    v = _tensor(torch.float8_e4m3fn, head_dim=256)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        assert _call(q, k, v) is False


def test_fp8_mma_hit_is_asserted_not_fallback():
    """Acceptance: the routing decision must be the FP8 kernel, not fallback."""
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    assert (
        _is_sm90_fp8_kv_mma(
            q, k, v, True, (None, None), device_capability_major=9
        )
        is True
    )


def test_gemma4_full_attention_d512_shape_routes_to_fp8():
    """Gemma-4 full attention: head_dim=512, FP8 KV, SM90, causal."""
    q = _tensor(torch.bfloat16, head_dim=512, heads=32)
    k = _tensor(torch.float8_e4m3fn, head_dim=512, heads=8)
    v = _tensor(torch.float8_e4m3fn, head_dim=512, heads=8)
    assert (
        _is_sm90_fp8_kv_mma(
            q, k, v, True, (None, None), device_capability_major=9
        )
        is True
    )


def test_gemma4_sliding_window_d256_stays_on_fa3():
    """Gemma-4 sliding-window layers (d256) must not take the d512 FP8 path."""
    q = _tensor(torch.bfloat16, head_dim=256, heads=32)
    k = _tensor(torch.float8_e4m3fn, head_dim=256, heads=8)
    v = _tensor(torch.float8_e4m3fn, head_dim=256, heads=8)
    assert (
        _is_sm90_fp8_kv_mma(
            q, k, v, True, (128, 0), device_capability_major=9
        )
        is False
    )


def test_sm90_fp8_kv_mma_explicit_major_override():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    assert (
        fa4.is_sm90_fp8_kv_mma(
            q, k, v, True, (None, None), device_capability_major=9
        )
        is True
    )
    assert (
        fa4.is_sm90_fp8_kv_mma(
            q, k, v, True, (None, None), device_capability_major=10
        )
        is False
    )
