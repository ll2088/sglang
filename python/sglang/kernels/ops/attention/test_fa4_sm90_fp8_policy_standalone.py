# Copyright (c) 2026, SGLang Team.
"""Standard-library-only checks for the FA4 SM90 FP8 routing policy.

Runs without torch, sglang, or a GPU so CI can validate routing before the
full dependency install succeeds.
"""

import importlib.util
import pathlib
import sys
import unittest

_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")
_spec = importlib.util.spec_from_file_location("_fa4_sm90_fp8_policy_std", _POLICY_PATH)
fa4 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fa4
_spec.loader.exec_module(fa4)


class _Dtype:
    def __init__(self, name):
        self.__name__ = name


class _Tensor:
    def __init__(self, dtype, head_dim):
        self.dtype = dtype
        self.shape = (1, 8, head_dim)


E4M3 = _Dtype("float8_e4m3fn")
E5M2 = _Dtype("float8_e5m2")
BF16 = _Dtype("bfloat16")


def _call(q, k, v, causal=True, window=(None, None), major=9):
    return fa4.is_sm90_fp8_kv_mma(
        q, k, v, causal, window, device_capability_major=major
    )


class RoutingTest(unittest.TestCase):
    def test_d512_fp8_kv_routes_to_mma(self):
        self.assertTrue(_call(_Tensor(BF16, 512), _Tensor(E4M3, 512), _Tensor(E4M3, 512)))

    def test_non_sm90_falls_back(self):
        self.assertFalse(_call(_Tensor(BF16, 512), _Tensor(E4M3, 512), _Tensor(E4M3, 512), major=10))

    def test_sm120_policy_unchanged(self):
        self.assertFalse(_call(_Tensor(BF16, 512), _Tensor(E4M3, 512), _Tensor(E4M3, 512), major=12))

    def test_non_d512_falls_back(self):
        self.assertFalse(_call(_Tensor(BF16, 256), _Tensor(E4M3, 256), _Tensor(E4M3, 256)))

    def test_non_fp8_kv_falls_back(self):
        self.assertFalse(_call(_Tensor(BF16, 512), _Tensor(BF16, 512), _Tensor(BF16, 512)))

    def test_e5m2_kv_falls_back(self):
        self.assertFalse(_call(_Tensor(BF16, 512), _Tensor(E5M2, 512), _Tensor(E5M2, 512)))

    def test_unequal_head_dims_fall_back(self):
        self.assertFalse(_call(_Tensor(BF16, 512), _Tensor(E4M3, 512), _Tensor(E4M3, 256)))

    def test_noncausal_full_attention_rejected(self):
        self.assertFalse(_call(_Tensor(BF16, 512), _Tensor(E4M3, 512), _Tensor(E4M3, 512), causal=False))

    def test_noncausal_sliding_window_allowed(self):
        self.assertTrue(_call(_Tensor(BF16, 512), _Tensor(E4M3, 512), _Tensor(E4M3, 512), causal=False, window=(128, 0)))

    def test_gemma4_sliding_window_d256_stays_on_fa3(self):
        self.assertFalse(_call(_Tensor(BF16, 256), _Tensor(E4M3, 256), _Tensor(E4M3, 256), window=(128, 0)))

    def test_stability_repeated_routing(self):
        args = (_Tensor(BF16, 512), _Tensor(E4M3, 512), _Tensor(E4M3, 512))
        self.assertEqual({_call(*args) for _ in range(3)}, {True})


class SplitTest(unittest.TestCase):
    def _splits(self, **kw):
        base = dict(fa_version=4, max_seqlen_k=4096, num_heads=8, num_kv_heads=1, num_sms=32, is_fp8_mma=True)
        base.update(kw)
        return fa4.select_num_splits(**base)

    def test_fa3_keeps_cap(self):
        self.assertEqual(self._splits(fa_version=3, is_fp8_mma=False), 32)

    def test_fa4_non_fp8_uses_heuristic(self):
        self.assertEqual(self._splits(is_fp8_mma=False), 0)

    def test_fa4_fp8_auto(self):
        self.assertEqual(self._splits(), 4)

    def test_short_seqlen_single_split(self):
        self.assertEqual(self._splits(max_seqlen_k=64), 1)

    def test_zero_seqlen_single_split(self):
        self.assertEqual(self._splits(max_seqlen_k=0), 1)

    def test_zero_sms_single_split(self):
        self.assertEqual(self._splits(num_sms=0), 1)

    def test_zero_heads_single_split(self):
        self.assertEqual(self._splits(num_heads=0), 1)

    def test_never_exceeds_tiles(self):
        self.assertEqual(self._splits(max_seqlen_k=256, num_sms=1024), 2)

    def test_stability_repeated_splits(self):
        self.assertEqual({self._splits() for _ in range(3)}, {4})


class DescaleTest(unittest.TestCase):
    def test_none_passthrough(self):
        self.assertIsNone(fa4.normalize_kv_descale(None, batch=2, heads=4))

    def test_scalar_broadcast(self):
        out = fa4.normalize_kv_descale(0.5, batch=2, heads=4)
        self.assertEqual(tuple(out.shape), (2, 4))

    def test_bad_1d_rejected(self):
        with self.assertRaises(ValueError):
            fa4.normalize_kv_descale(object(), batch=2, heads=4)


if __name__ == "__main__":
    unittest.main()
