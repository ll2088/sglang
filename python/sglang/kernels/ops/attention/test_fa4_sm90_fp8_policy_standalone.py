# Copyright (c) 2026, SGLang Team.
"""Zero-dependency entry point for the FA4 SM90 FP8 routing policy.

This file only needs pytest: it loads ``fa4_sm90_fp8_policy.py`` by path and
never imports the sglang package, torch, or the FA4 extension. CI can run it
in a lightweight job without installing the full sglang dependency set.
"""

import importlib.util
import pathlib

_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")


def _load_policy():
    spec = importlib.util.spec_from_file_location(
        "_fa4_sm90_fp8_policy_standalone", _POLICY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_imports_without_torch_or_sglang():
    policy = _load_policy()
    assert policy.SM90_FP8_MMA_HEAD_DIM == 512
    assert policy.SM90_FP8_MMA_MAX_SPLITS == 4


def test_split_selection_is_deterministic():
    policy = _load_policy()
    kwargs = dict(
        fa_version=4,
        max_seqlen_k=4096,
        num_heads=32,
        num_kv_heads=8,
        num_sms=132,
        is_fp8_mma=True,
    )
    first = policy.select_num_splits(**kwargs)
    assert first == policy.select_num_splits(**kwargs)
    assert 1 <= first <= policy.SM90_FP8_MMA_MAX_SPLITS


def test_fa3_split_cap_is_retained():
    policy = _load_policy()
    assert (
        policy.select_num_splits(
            fa_version=3,
            max_seqlen_k=4096,
            num_heads=32,
            num_kv_heads=8,
            num_sms=132,
            is_fp8_mma=False,
        )
        == 32
    )
