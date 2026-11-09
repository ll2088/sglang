# Copyright (c) 2026, SGLang Team.
"""Standalone checks for the FA4 SM90 FP8 policy module.

Runs without sglang, CUDA, or a compiled extension: only the pure policy
module is loaded by path. This is the lightweight CI entry point that is not
blocked by the heavy `pip install -e python` dependency chain.
"""

import importlib.util
import pathlib
import sys

import pytest

_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")


def _load_policy():
    spec = importlib.util.spec_from_file_location(
        "_fa4_sm90_fp8_policy_standalone", _POLICY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover
        sys.modules.pop(spec.name, None)
        raise
    return module


def test_policy_imports_without_torch_or_sglang():
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch blocked for test")
        if name == "sglang" or name.startswith("sglang."):
            raise ImportError("sglang blocked for test")
        return real_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(builtins, "__import__", _blocked)
        mod = _load_policy()
    assert mod.SM90_FP8_MMA_HEAD_DIM == 512


def test_split_selection_fa3_cap_and_fa4_auto():
    mod = _load_policy()
    assert (
        mod.select_num_splits(
            fa_version=3,
            max_seqlen_k=4096,
            num_heads=8,
            num_kv_heads=1,
            num_sms=32,
            is_fp8_mma=False,
        )
        == 32
    )
    assert (
        mod.select_num_splits(
            fa_version=4,
            max_seqlen_k=4096,
            num_heads=8,
            num_kv_heads=1,
            num_sms=32,
            is_fp8_mma=True,
        )
        == 4
    )
    assert (
        mod.select_num_splits(
            fa_version=4,
            max_seqlen_k=64,
            num_heads=8,
            num_kv_heads=1,
            num_sms=32,
            is_fp8_mma=True,
        )
        == 1
    )


def test_split_selection_degenerate_inputs():
    mod = _load_policy()
    for kwargs in (
        dict(max_seqlen_k=0, num_heads=8, num_kv_heads=1, num_sms=32),
        dict(max_seqlen_k=4096, num_heads=8, num_kv_heads=1, num_sms=0),
        dict(max_seqlen_k=4096, num_heads=0, num_kv_heads=1, num_sms=32),
    ):
        assert (
            mod.select_num_splits(fa_version=4, is_fp8_mma=True, **kwargs) == 1
        )


def test_split_selection_never_exceeds_tiles():
    mod = _load_policy()
    assert (
        mod.select_num_splits(
            fa_version=4,
            max_seqlen_k=256,
            num_heads=8,
            num_kv_heads=1,
            num_sms=1024,
            is_fp8_mma=True,
        )
        == 1
    )
