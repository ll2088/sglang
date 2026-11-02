# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for the batched FP4 fake-quant layout.

These tests do not import Triton or require CUDA, so they can run in a
lightweight CI job even when the GPU environment is unavailable. They cover
the int64 block-offset arithmetic and the tail-mask batching that the Triton
kernel mirrors.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.quantization import fp4_fake_quant


@pytest.mark.parametrize("numel", [0, 1, 15, 16, 17, 32, 48, 1024])
@pytest.mark.parametrize("blocks_per_program", [1, 2, 32, 64, 128])
def test_batched_matches_per_block_cpu(numel, blocks_per_program):
    """Batched-block processing must be bit-identical to per-block on CPU."""
    torch.manual_seed(0)
    x = torch.randn(numel, dtype=torch.bfloat16)
    ref = fp4_fake_quant.fake_quant_fp4(x)
    got = fp4_fake_quant.fake_quant_fp4_batched(
        x, blocks_per_program=blocks_per_program
    )
    assert torch.equal(got, ref)


def test_batched_tail_mask_cpu():
    """A non-divisible block count exercises the tail mask."""
    torch.manual_seed(1)
    # 100 blocks of 16 elements is not divisible by 64.
    x = torch.randn(100 * 16, dtype=torch.bfloat16)
    ref = fp4_fake_quant.fake_quant_fp4(x)
    got = fp4_fake_quant.fake_quant_fp4_batched(x, blocks_per_program=64)
    assert torch.equal(got, ref)


@pytest.mark.parametrize("num_blocks", [1, 2, 63, 64, 65, 100, 129])
@pytest.mark.parametrize("blocks_per_program", [1, 2, 32, 64, 128])
def test_batched_tail_mask_all_groupings_cpu(num_blocks, blocks_per_program):
    """Every grouping must reproduce per-block output across tail boundaries."""
    torch.manual_seed(4)
    x = torch.randn(num_blocks * 16, dtype=torch.bfloat16)
    ref = fp4_fake_quant.fake_quant_fp4(x)
    got = fp4_fake_quant.fake_quant_fp4_batched(
        x, blocks_per_program=blocks_per_program
    )
    assert torch.equal(got, ref)


def test_int64_offset_arithmetic_cpu():
    """Block offsets must be int64 and exceed int32 range past 2**31 elements."""
    num_blocks = 2**31 // 16 + 1024
    offsets = fp4_fake_quant._block_offsets(num_blocks, 64, "cpu")
    assert offsets.dtype == torch.int64
    assert offsets.numel() == num_blocks
    assert offsets[0].item() == 0
    assert offsets[-1].item() == num_blocks - 1
    last_element = offsets[-1].item() * 16
    assert last_element > 2**31


def test_int64_offset_arithmetic_grouped_cpu():
    """Grouped offsets must match the flat int64 layout for any grouping."""
    num_blocks = 2**31 // 16 + 1024
    flat = fp4_fake_quant._block_offsets(num_blocks, 1, "cpu")
    for blocks_per_program in (2, 32, 64, 128):
        grouped = fp4_fake_quant._block_offsets(
            num_blocks, blocks_per_program, "cpu"
        )
        assert grouped.dtype == torch.int64
        assert torch.equal(grouped, flat)


def test_invalid_blocks_per_program_offsets_cpu():
    """Non-positive grouping must be rejected by the offset helper."""
    with pytest.raises(ValueError):
        fp4_fake_quant._block_offsets(16, 0, "cpu")


def test_invalid_blocks_per_program_cpu():
    """Non-positive batching must be rejected rather than silently misbehave."""
    x = torch.randn(64, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fp4_fake_quant.fake_quant_fp4_batched(x, blocks_per_program=0)


def test_exported_symbols_cpu():
    """The quantization package must export the fake-quant entry points."""
    from sglang.kernels.ops import quantization

    assert "fp4_fake_quant" in quantization.__all__
    assert quantization.fp4_fake_quant is fp4_fake_quant
    for name in (
        "fake_quant_fp4",
        "fake_quant_fp4_batched",
        "fake_quant_compressed_kv",
        "STATIC_FP4_BLOCKS_PER_PROGRAM",
    ):
        assert hasattr(quantization.fp4_fake_quant, name)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
