# Copyright (c) 2026, SGLang Team.
"""Standalone routing checks for the FA4 SM90 d512 FP8-MMA policy.

Runs with the standard library only (no sglang install, no torch, no GPU):

    PYTHONPATH=python python python/sglang/kernels/ops/attention/run_fa4_fp8_routing_tests.py
"""

import importlib.util
import pathlib
import sys

_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")


def _load_policy():
    spec = importlib.util.spec_from_file_location(
        "_fa4_sm90_fp8_policy_standalone", _POLICY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeTensor:
    def __init__(self, dtype_name, head_dim):
        self.dtype = type(dtype_name, (), {"__name__": dtype_name})
        self.shape = (1, 8, head_dim)


E4M3 = "float8_e4m3fn"
E5M2 = "float8_e5m2"
BF16 = "bfloat16"


def _q(head_dim=512, dtype=BF16):
    return _FakeTensor(dtype, head_dim)


def _kv(head_dim=512, dtype=E4M3):
    return _FakeTensor(dtype, head_dim), _FakeTensor(dtype, head_dim)


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_sm90_d512_fp8_routes_to_mma(policy):
    k, v = _kv()
    _check(
        policy.is_sm90_fp8_kv_mma(_q(), k, v, True, (None, None), 9) is True,
        "SM90 d512 FP8 KV causal must route to FP8 MMA",
    )


def test_non_sm90_falls_back(policy):
    k, v = _kv()
    _check(
        policy.is_sm90_fp8_kv_mma(_q(), k, v, True, (None, None), 10) is False,
        "non-SM90 must fall back",
    )


def test_non_d512_falls_back(policy):
    k, v = _kv(head_dim=256)
    _check(
        policy.is_sm90_fp8_kv_mma(_q(head_dim=256), k, v, True, (None, None), 9)
        is False,
        "d256 must fall back",
    )


def test_non_fp8_kv_falls_back(policy):
    k, v = _kv(dtype=BF16)
    _check(
        policy.is_sm90_fp8_kv_mma(_q(), k, v, True, (None, None), 9) is False,
        "BF16 KV must fall back",
    )


def test_e5m2_kv_falls_back(policy):
    k, v = _kv(dtype=E5M2)
    _check(
        policy.is_sm90_fp8_kv_mma(_q(), k, v, True, (None, None), 9) is False,
        "E5M2 KV must fall back",
    )


def test_unequal_head_dims_fall_back(policy):
    k = _FakeTensor(E4M3, 512)
    v = _FakeTensor(E4M3, 256)
    _check(
        policy.is_sm90_fp8_kv_mma(_q(), k, v, True, (None, None), 9) is False,
        "unequal head dims must fall back",
    )


def test_noncausal_full_attention_rejected(policy):
    k, v = _kv()
    _check(
        policy.is_sm90_fp8_kv_mma(_q(), k, v, False, (None, None), 9) is False,
        "noncausal full attention must be rejected",
    )


def test_noncausal_sliding_window_allowed(policy):
    k, v = _kv()
    _check(
        policy.is_sm90_fp8_kv_mma(_q(), k, v, False, (128, 0), 9) is True,
        "noncausal sliding window must be allowed",
    )


def test_sm120_policy_unchanged(policy):
    k, v = _kv()
    _check(
        policy.is_sm90_fp8_kv_mma(_q(), k, v, True, (None, None), 12) is False,
        "SM120 must not take the SM90 FP8-MMA gate",
    )


def test_split_auto_selection(policy):
    cases = [
        (dict(fa_version=3, is_fp8_mma=False), 32),
        (dict(fa_version=3, is_fp8_mma=True), 32),
        (dict(fa_version=4, is_fp8_mma=False), 0),
        (dict(fa_version=4, is_fp8_mma=True), 4),
    ]
    for extra, expected in cases:
        got = policy.select_num_splits(
            max_seqlen_k=4096,
            num_heads=8,
            num_kv_heads=1,
            num_sms=32,
            **extra,
        )
        _check(got == expected, f"split {extra} expected {expected}, got {got}")


def test_split_short_seqlen(policy):
    got = policy.select_num_splits(
        fa_version=4,
        max_seqlen_k=64,
        num_heads=8,
        num_kv_heads=1,
        num_sms=32,
        is_fp8_mma=True,
    )
    _check(got == 1, f"short seqlen expected 1 split, got {got}")


def test_split_degenerate_inputs(policy):
    for kwargs in (
        dict(max_seqlen_k=0, num_heads=8, num_sms=32),
        dict(max_seqlen_k=4096, num_heads=8, num_sms=0),
        dict(max_seqlen_k=4096, num_heads=0, num_sms=32),
    ):
        got = policy.select_num_splits(
            fa_version=4, num_kv_heads=1, is_fp8_mma=True, **kwargs
        )
        _check(got == 1, f"degenerate {kwargs} expected 1 split, got {got}")


def test_split_never_exceeds_tiles(policy):
    got = policy.select_num_splits(
        fa_version=4,
        max_seqlen_k=256,
        num_heads=8,
        num_kv_heads=1,
        num_sms=1024,
        is_fp8_mma=True,
    )
    _check(got == 4, f"expected 4 splits bounded by tiles, got {got}")


def test_policy_module_is_torch_free(policy):
    _check("torch" not in policy.__dict__, "policy module must not import torch")
    _check("sglang" not in policy.__dict__, "policy module must not import sglang")


def main() -> int:
    policy = _load_policy()
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for test in tests:
        try:
            test(policy)
        except AssertionError as exc:
            failures.append((test.__name__, str(exc)))
    for name, msg in failures:
        print(f"FAIL {name}: {msg}")
    print(f"{len(tests) - len(failures)}/{len(tests)} routing checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
