# Copyright (c) 2026, SGLang Team.
"""Standalone checks for the FA4 SM90 d512 FP8-MMA routing policy.

Runs with the standard library only: no torch, no sglang, no GPU. This is the
entry point CI should use for the routing/descale/split dimensions so that the
environment setup never needs to install sglang.
"""

import importlib.util
import pathlib
import sys
import unittest

_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")


def _load_policy():
    name = "_fa4_sm90_fp8_policy_standalone"
    spec = importlib.util.spec_from_file_location(name, _POLICY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


policy = _load_policy()


class _FakeTensor:
    def __init__(self, dtype_name, head_dim):
        self.dtype = type(dtype_name, (), {"__name__": dtype_name})
        self.shape = (1, 8, head_dim)


BF16 = _FakeTensor("bfloat16", 512)
FP16 = _FakeTensor("float16", 512)
FP8_K = _FakeTensor("float8_e4m3fn", 512)
FP8_V = _FakeTensor("float8_e4m3fn", 512)
FP8_E5M2 = _FakeTensor("float8_e5m2", 512)
D256_K = _FakeTensor("float8_e4m3fn", 256)
D256_V = _FakeTensor("float8_e4m3fn", 256)
D256_Q = _FakeTensor("bfloat16", 256)


class RoutingTest(unittest.TestCase):
    def test_sm90_d512_fp8_routes_to_mma(self):
        for q in (BF16, FP16):
            self.assertTrue(
                policy.is_sm90_fp8_kv_mma(
                    q, FP8_K, FP8_V, True, (None, None),
                    device_capability_major=9,
                )
            )

    def test_non_sm90_falls_back(self):
        self.assertFalse(
            policy.is_sm90_fp8_kv_mma(
                BF16, FP8_K, FP8_V, True, (None, None),
                device_capability_major=10,
            )
        )

    def test_sm120_policy_unchanged(self):
        self.assertFalse(
            policy.is_sm90_fp8_kv_mma(
                BF16, FP8_K, FP8_V, True, (None, None),
                device_capability_major=12,
            )
        )

    def test_non_d512_falls_back(self):
        self.assertFalse(
            policy.is_sm90_fp8_kv_mma(
                D256_Q, D256_K, D256_V, True, (128, 0),
                device_capability_major=9,
            )
        )

    def test_non_fp8_kv_falls_back(self):
        self.assertFalse(
            policy.is_sm90_fp8_kv_mma(
                BF16, BF16, BF16, True, (None, None),
                device_capability_major=9,
            )
        )

    def test_e5m2_kv_falls_back(self):
        self.assertFalse(
            policy.is_sm90_fp8_kv_mma(
                BF16, FP8_E5M2, FP8_E5M2, True, (None, None),
                device_capability_major=9,
            )
        )

    def test_noncausal_full_attention_rejected(self):
        with self.assertRaises(NotImplementedError):
            policy.is_sm90_fp8_kv_mma(
                BF16, FP8_K, FP8_V, False, (None, None),
                device_capability_major=9,
            )

    def test_noncausal_sliding_window_allowed(self):
        self.assertTrue(
            policy.is_sm90_fp8_kv_mma(
                BF16, FP8_K, FP8_V, False, (128, 0),
                device_capability_major=9,
            )
        )

    def test_gemma4_full_attention_d512_routes_to_fp8(self):
        self.assertTrue(
            policy.is_sm90_fp8_kv_mma(
                BF16, FP8_K, FP8_V, True, (None, None),
                device_capability_major=9,
            )
        )

    def test_gemma4_sliding_window_d256_stays_on_fa3(self):
        self.assertFalse(
            policy.is_sm90_fp8_kv_mma(
                D256_Q, D256_K, D256_V, True, (128, 0),
                device_capability_major=9,
            )
        )

    def test_repeated_routing_is_stable(self):
        results = {
            policy.is_sm90_fp8_kv_mma(
                BF16, FP8_K, FP8_V, True, (None, None),
                device_capability_major=9,
            )
            for _ in range(3)
        }
        self.assertEqual(results, {True})


class SplitTest(unittest.TestCase):
    def _splits(self, **kwargs):
        base = dict(
            fa_version=4,
            max_seqlen_k=4096,
            num_heads=8,
            num_kv_heads=1,
            num_sms=32,
            is_fp8_mma=True,
        )
        base.update(kwargs)
        return policy.select_num_splits(**base)

    def test_fa3_keeps_cap(self):
        self.assertEqual(self._splits(fa_version=3, is_fp8_mma=False), 32)
        self.assertEqual(self._splits(fa_version=3, is_fp8_mma=True), 32)

    def test_fa4_non_fp8_uses_kernel_heuristic(self):
        self.assertEqual(self._splits(is_fp8_mma=False), 0)

    def test_fa4_fp8_auto_selects(self):
        self.assertEqual(self._splits(), 4)

    def test_short_seqlen_uses_one_split(self):
        self.assertEqual(self._splits(max_seqlen_k=64), 1)

    def test_zero_seqlen_uses_one_split(self):
        self.assertEqual(self._splits(max_seqlen_k=0), 1)

    def test_zero_sms_uses_one_split(self):
        self.assertEqual(self._splits(num_sms=0), 1)

    def test_zero_heads_uses_one_split(self):
        self.assertEqual(self._splits(num_heads=0), 1)

    def test_never_exceeds_tiles(self):
        self.assertEqual(self._splits(max_seqlen_k=256, num_sms=1024), 1)

    def test_repeated_selection_is_stable(self):
        self.assertEqual({self._splits() for _ in range(3)}, {4})


class DescaleTest(unittest.TestCase):
    def test_none_passthrough(self):
        self.assertIsNone(policy.normalize_kv_descale(None, batch=2, heads=4))

    def test_python_scalar(self):
        out = policy.normalize_kv_descale(0.5, batch=2, heads=4)
        self.assertEqual(tuple(out.shape), (2, 4))
        self.assertEqual(out.dtype.__name__, "float32")

    def test_bad_1d_rejected(self):
        with self.assertRaises(ValueError):
            policy.normalize_kv_descale(object(), batch=2, heads=4)

    def test_policy_module_has_no_torch_or_sglang(self):
        self.assertNotIn("torch", policy.__dict__)
        self.assertNotIn("sglang", policy.__dict__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
