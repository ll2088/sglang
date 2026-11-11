"""CPU guards for the MegaMoE fused-shared-expert weight contract.

Runs without the sglang package installed: the requant helper is loaded by
path from mxfp8_block_convert.py, and the fused-shared-expert layout is
reproduced with plain torch so the shape/dtype contract and the numerical
agreement with a reference block-FP8 path can be checked on CPU.
"""

import importlib.util
import pathlib
import unittest

import torch

# Self-contained: load the requant helper by path so this guard runs in a
# torch+pytest-only environment (no sglang install, no pip resolution).
_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[6]
    / "python/sglang/srt/layers/quantization/mxfp8_block_convert.py"
)
assert _MODULE_PATH.is_file(), f"module under test not found: {_MODULE_PATH}"
_spec = importlib.util.spec_from_file_location(
    "_mxfp8_block_convert_under_test", _MODULE_PATH
)
assert _spec is not None and _spec.loader is not None
_mxfp8_block_convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mxfp8_block_convert)
requant_block_fp8_to_mxfp8 = _mxfp8_block_convert.requant_block_fp8_to_mxfp8
MXFP8_BLOCK_SIZE = _mxfp8_block_convert.MXFP8_BLOCK_SIZE
assert "sglang" not in _mxfp8_block_convert.__dict__.get("__name__", "")

FP8 = torch.float8_e4m3fn
BLOCK = 128


def _block_fp8(real: torch.Tensor):
    """Quantize ``[N, K]`` to E4M3 with one fp32 scale per 128x128 block."""
    n, k = real.shape
    blocks = real.reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=(1, 3)) / 448.0
    full = scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return (real / full).to(FP8), scale


def _dequant_ue8m0(weight: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    return weight.float() * torch.exp2(sf.float() - 127).repeat_interleave(
        MXFP8_BLOCK_SIZE, -1
    )


def _fused_shared_layout(
    routed_w13: torch.Tensor,
    routed_s13: torch.Tensor,
    shared_w13: torch.Tensor,
    shared_s13: torch.Tensor,
):
    """Concatenate routed and shared L1 shards the way the fused kernel expects.

    The fused shared expert is appended as extra rows of the same L1 weight and
    scale tensors, so the kernel sees one contiguous expert table.
    """
    w13 = torch.cat([routed_w13, shared_w13], dim=0)
    s13 = torch.cat([routed_s13, shared_s13], dim=0)
    return w13, s13


class TestMegaMoEFusedSharedContract(unittest.TestCase):
    def test_fused_shared_matches_reference_within_tolerance(self):
        torch.manual_seed(0)
        routed_real = torch.randn(2 * BLOCK, 2 * BLOCK) * 0.05
        shared_real = torch.randn(BLOCK, 2 * BLOCK) * 0.05
        routed_w, routed_s = _block_fp8(routed_real)
        shared_w, shared_s = _block_fp8(shared_real)

        fused_w, fused_s = _fused_shared_layout(
            routed_w, routed_s, shared_w, shared_s
        )
        requant_w, requant_sf = requant_block_fp8_to_mxfp8(fused_w, fused_s)

        ref = _dequant_ue8m0(requant_w, requant_sf)
        # Reference: dequantize each shard with its own block scale, then stack.
        ref_routed = routed_w.float() * routed_s.repeat_interleave(
            BLOCK, 0
        ).repeat_interleave(BLOCK, 1)
        ref_shared = shared_w.float() * shared_s.repeat_interleave(
            BLOCK, 0
        ).repeat_interleave(BLOCK, 1)
        ref_expected = torch.cat([ref_routed, ref_shared], dim=0)

        self.assertEqual(requant_w.shape, fused_w.shape)
        self.assertEqual(requant_sf.shape, (fused_w.shape[0], fused_w.shape[1] // 32))
        self.assertTrue(torch.isfinite(ref).all())
        rel = (ref - ref_expected).abs() / ref_expected.abs().clamp(min=1e-6)
        self.assertLess(rel.max().item(), 0.1)

    def test_empty_expert_batch_contract(self):
        weight = torch.zeros((0, 128), dtype=FP8)
        scale = torch.ones((0, 1), dtype=torch.float32)
        requant, sf = requant_block_fp8_to_mxfp8(weight, scale)
        self.assertEqual(requant.shape, (0, 128))
        self.assertEqual(requant.dtype, FP8)
        self.assertEqual(sf.shape, (0, 128 // MXFP8_BLOCK_SIZE))
        self.assertEqual(sf.dtype, torch.uint8)

    def test_single_token_contract(self):
        weight = torch.zeros((1, 128), dtype=FP8)
        scale = torch.ones((1, 1), dtype=torch.float32)
        requant, sf = requant_block_fp8_to_mxfp8(weight, scale)
        self.assertEqual(requant.shape, (1, 128))
        self.assertEqual(sf.shape, (1, 128 // MXFP8_BLOCK_SIZE))
        self.assertTrue(torch.isfinite(requant.float()).all())

    def test_large_batch_shape_contract(self):
        weight = torch.zeros((1024, 512), dtype=FP8)
        scale = torch.ones((8, 4), dtype=torch.float32)
        requant, sf = requant_block_fp8_to_mxfp8(weight, scale)
        self.assertEqual(requant.shape, (1024, 512))
        self.assertEqual(requant.dtype, FP8)
        self.assertEqual(sf.shape, (1024, 512 // MXFP8_BLOCK_SIZE))
        self.assertEqual(sf.dtype, torch.uint8)
        self.assertTrue(torch.isfinite(requant.float()).all())

    def test_scale_saturation_does_not_produce_nan(self):
        weight = torch.full((128, 128), 448.0, dtype=FP8)
        scale = torch.full((1, 1), 1e30, dtype=torch.float32)
        requant, sf = requant_block_fp8_to_mxfp8(weight, scale)
        self.assertTrue(torch.isfinite(requant.float()).all())
        self.assertTrue((sf <= 254).all())


if __name__ == "__main__":
    unittest.main(verbosity=3)
