# Copyright (c) 2026, SGLang Team.
"""Minimal-environment checks for the FA4 SM90 FP8 routing policy.

Runs with only pytest installed: no torch, no sglang package import.
"""

import importlib.util
import pathlib
import sys

import pytest

_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")
_NAME = "_fa4_sm90_fp8_policy_standalone"
_spec = importlib.util.spec_from_file_location(_NAME, _POLICY_PATH)
policy = importlib.util.module_from_spec(_spec)
sys.modules[_NAME] = policy
_spec.loader.exec_module(policy)


def test_module_imports_without_torch_or_sglang():
    assert "torch" not in policy.__dict__
    assert "sglang" not in policy.__dict__
    assert policy.SM90_FP8_MMA_HEAD_DIM == 512


def test_descale_none_passthrough():
    assert policy.normalize_kv_descale(None, batch=2, heads=4) is None


def test_split_fa3_cap_and_fa4_auto():
    assert policy.select_num_splits(
        fa_version=3, max_seqlen_k=4096, num_heads=8, num_kv_heads=1,
        num_sms=32, is_fp8_mma=False,
    ) == 32
    assert policy.select_num_splits(
        fa_version=4, max_seqlen_k=4096, num_heads=8, num_kv_heads=1,
        num_sms=32, is_fp8_mma=False,
    ) == 0


def test_split_degenerate_inputs():
    for kwargs in (
        dict(max_seqlen_k=0, num_sms=32),
        dict(max_seqlen_k=4096, num_sms=0),
        dict(max_seqlen_k=4096, num_sms=32, num_heads=0),
    ):
        assert policy.select_num_splits(
            fa_version=4, num_heads=8, num_kv_heads=1, is_fp8_mma=True, **kwargs
        ) == 1


def test_routing_without_torch_falls_back():
    class _T:
        dtype = object()
        shape = (1, 1, 512)

    assert policy.is_sm90_fp8_kv_mma(
        _T(), _T(), _T(), True, (None, None), device_capability_major=9
    ) is False


def test_routing_non_sm90_falls_back():
    class _T:
        dtype = object()
        shape = (1, 1, 512)

    assert policy.is_sm90_fp8_kv_mma(
        _T(), _T(), _T(), True, (None, None), device_capability_major=10
    ) is False
