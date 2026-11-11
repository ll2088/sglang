"""CPU guards for the MXFP8 dispatch cache hit/invalidate contract.

Runs without the sglang package installed: the module under test is loaded by
path and FlashInfer is stubbed out.
"""

import importlib.util
import pathlib
import unittest

import torch

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[6]
    / "python/sglang/srt/layers/quantization/mxfp8_dispatch_cache.py"
)
assert _MODULE_PATH.is_file(), f"module under test not found: {_MODULE_PATH}"
_spec = importlib.util.spec_from_file_location(
    "_mxfp8_dispatch_cache_under_test", _MODULE_PATH
)
assert _spec is not None and _spec.loader is not None
_mxfp8_dispatch_cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mxfp8_dispatch_cache)
maybe_cache_mxfp8_dispatch = _mxfp8_dispatch_cache.maybe_cache_mxfp8_dispatch


class _FakeGemmBase:
    """Stand-in for flashinfer.gemm.gemm_base with a stateless runner factory."""

    def __init__(self):
        self.calls = 0

        def factory(*args, **kwargs):
            self.calls += 1
            return object()

        self._cute_dsl_gemm_mxfp8_runner = factory


class TestMxfp8DispatchCache(unittest.TestCase):
    def setUp(self):
        self.gemm_base = _FakeGemmBase()
        self.raw_calls = []

        def raw_mm(a, b, a_scale, b_scale, out_dtype, use_8x4_sf_layout, backend, skip_check=False):
            self.raw_calls.append(skip_check)
            return "out"

        self.raw_mm = raw_mm

    def _wrap(self):
        # Patch the module-level gemm_base and version so the cache is built.
        _mxfp8_dispatch_cache.gemm_base = self.gemm_base
        _mxfp8_dispatch_cache.flashinfer_version = "0.7.0"
        return maybe_cache_mxfp8_dispatch(self.raw_mm)

    def _tensors(self, m, k):
        a = torch.zeros((m, k), dtype=torch.float8_e4m3fn)
        b = torch.zeros((k, k), dtype=torch.float8_e4m3fn)
        a_scale = torch.zeros((m, k // 32), dtype=torch.uint8)
        b_scale = torch.zeros((k, k // 32), dtype=torch.uint8)
        return a, b, a_scale, b_scale

    def test_unsupported_version_returns_raw(self):
        _mxfp8_dispatch_cache.gemm_base = self.gemm_base
        _mxfp8_dispatch_cache.flashinfer_version = "0.6.0"
        self.assertIs(maybe_cache_mxfp8_dispatch(self.raw_mm), self.raw_mm)

    def test_repeated_call_hits_cache(self):
        cached = self._wrap()
        args = self._tensors(64, 128)
        cached(*args, torch.bfloat16, True, "cute-dsl")
        cached(*args, torch.bfloat16, True, "cute-dsl")
        # First call validates, second skips validation.
        self.assertEqual(self.raw_calls, [False, True])

    def test_shape_change_invalidates(self):
        cached = self._wrap()
        cached(*self._tensors(64, 128), torch.bfloat16, True, "cute-dsl")
        cached(*self._tensors(128, 128), torch.bfloat16, True, "cute-dsl")
        self.assertEqual(self.raw_calls, [False, False])

    def test_non_cute_dsl_backend_bypasses_cache(self):
        cached = self._wrap()
        args = self._tensors(64, 128)
        cached(*args, torch.bfloat16, True, "cutlass")
        cached(*args, torch.bfloat16, True, "cutlass")
        self.assertEqual(self.raw_calls, [False, False])


if __name__ == "__main__":
    unittest.main(verbosity=3)
