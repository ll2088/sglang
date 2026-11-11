"""CPU guards for block-FP8 -> MXFP8 (UE8M0) weight requantization."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import importlib.util
import pathlib
import unittest

import torch

# Import the module by path so this CPU guard runs without installing sglang
# (the previous CI setup failed during pip resolution before pytest started).
_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[6]
    / "python/sglang/srt/layers/quantization/mxfp8_block_convert.py"
)
_spec = importlib.util.spec_from_file_location(
    "_mxfp8_block_convert_under_test", _MODULE_PATH
)
assert _spec is not None and _spec.loader is not None
_mxfp8_block_convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mxfp8_block_convert)
MXFP8_BLOCK_SIZE = _mxfp8_block_convert.MXFP8_BLOCK_SIZE
requant_block_fp8_to_mxfp8 = _mxfp8_block_convert.requant_block_fp8_to_mxfp8


class CustomTestCase(unittest.TestCase):
    """Minimal stand-in so the guard does not import sglang.test.test_utils."""

FP8 = torch.float8_e4m3fn
BLOCK = 128


def _block_fp8(real: torch.Tensor, power_of_two: bool = False):
    """Quantize ``[N, K]`` to E4M3 with one fp32 scale per 128x128 block."""
    n, k = real.shape
    blocks = real.reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=(1, 3)) / 448.0
    if power_of_two:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    full = scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return (real / full).to(FP8), scale


def _dequant_ue8m0(weight: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    return weight.float() * torch.exp2(sf.float() - 127).repeat_interleave(
        MXFP8_BLOCK_SIZE, -1
    )


class TestBlockFp8ToMxfp8Requant(CustomTestCase):
    def test_exact_for_power_of_two_block_scales(self):
        torch.manual_seed(0)
        real = torch.randn(256, 256) * 0.05
        weight, scale = _block_fp8(real, power_of_two=True)

        requant, sf = requant_block_fp8_to_mxfp8(weight, scale)

        self.assertEqual(requant.dtype, FP8)
        self.assertEqual(sf.dtype, torch.uint8)
        self.assertEqual(sf.shape, (256, 256 // MXFP8_BLOCK_SIZE))
        # A finer power-of-two scale only shifts exponents, so nothing rounds.
        torch.testing.assert_close(
            _dequant_ue8m0(requant, sf),
            weight.float()
            * scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1),
            rtol=0.0,
            atol=0.0,
        )

    def test_rounding_error_bounded_for_arbitrary_scales(self):
        torch.manual_seed(1)
        real = torch.randn(256, 256) * 0.05
        weight, scale = _block_fp8(real)

        requant, sf = requant_block_fp8_to_mxfp8(weight, scale)

        deq = _dequant_ue8m0(requant, sf)
        ref = weight.float() * scale.repeat_interleave(BLOCK, 0).repeat_interleave(
            BLOCK, 1
        )
        self.assertTrue(torch.isfinite(deq).all())
        # One extra power-of-two rounding step: relative error stays small.
        rel = (deq - ref).abs() / ref.abs().clamp(min=1e-6)
        self.assertLess(rel.max().item(), 0.1)

    def test_scale_saturation_does_not_produce_nan(self):
        weight = torch.full((128, 128), 448.0, dtype=FP8)
        scale = torch.full((1, 1), 1e30, dtype=torch.float32)

        requant, sf = requant_block_fp8_to_mxfp8(weight, scale)

        self.assertTrue(torch.isfinite(requant.float()).all())
        self.assertTrue((sf <= 254).all())

    def test_rejects_unaligned_k(self):
        weight = torch.zeros((128, 96), dtype=FP8)
        scale = torch.ones((1, 1), dtype=torch.float32)
        with self.assertRaises(ValueError):
            requant_block_fp8_to_mxfp8(weight, scale)

    def test_rejects_scale_shape_that_does_not_cover_weight(self):
        weight = torch.zeros((256, 256), dtype=FP8)
        scale = torch.ones((1, 1), dtype=torch.float32)
        with self.assertRaises(ValueError):
            requant_block_fp8_to_mxfp8(weight, scale)

    def test_non_finite_inputs_stay_finite(self):
        weight = torch.full((128, 128), 448.0, dtype=FP8)
        scale = torch.full((1, 1), float("inf"), dtype=torch.float32)

        requant, sf = requant_block_fp8_to_mxfp8(weight, scale)

        self.assertTrue(torch.isfinite(requant.float()).all())
        self.assertTrue((sf <= 254).all())

    def test_empty_and_single_token_shapes(self):
        # Empty expert batch: zero rows must still produce a valid empty tensor.
        weight = torch.zeros((0, 128), dtype=FP8)
        scale = torch.ones((0, 1), dtype=torch.float32)
        requant, sf = requant_block_fp8_to_mxfp8(weight, scale)
        self.assertEqual(requant.shape, (0, 128))
        self.assertEqual(sf.shape, (0, 128 // MXFP8_BLOCK_SIZE))

        # Single token / single row.
        weight = torch.zeros((1, 128), dtype=FP8)
        scale = torch.ones((1, 1), dtype=torch.float32)
        requant, sf = requant_block_fp8_to_mxfp8(weight, scale)
        self.assertEqual(requant.shape, (1, 128))
        self.assertEqual(sf.shape, (1, 128 // MXFP8_BLOCK_SIZE))
        self.assertTrue(torch.isfinite(requant.float()).all())


if __name__ == "__main__":
    unittest.main(verbosity=3)
