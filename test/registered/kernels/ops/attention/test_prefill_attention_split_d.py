"""Split-D head-dim blocking and stride-alignment hint for prefill attention.

Covers the gfx1151-only path added to
``sglang.kernels.ops.attention.prefill_attention``: the split-D tail block,
the ``tl.multiple_of`` head-stride hint, and the conditional launch config.
The non-gfx1151 path must stay bit-identical to the single-block kernel.
"""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention import prefill_attention as prefill_ops
from sglang.kernels.ops.attention.prefill_attention import context_attention_fwd
from sglang.test.test_utils import CustomTestCase


def _cuda_available() -> bool:
    """Probe lazily so importing this module never touches the driver."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


# Resolved at import time but guarded: on a CPU-only runner every device test
# is skipped instead of failing during collection.
_CUDA_AVAILABLE = _cuda_available()
_DEVICE_SKIP = unittest.skipUnless(
    _CUDA_AVAILABLE, "requires a CUDA/HIP device"
)


class _LaunchCapture:
    """Records the launch in place of ``_fwd_kernel``."""

    args: tuple
    kwargs: dict

    def __getitem__(self, grid):
        return self._record

    def _record(self, *args, **kwargs) -> None:
        self.args, self.kwargs = args, kwargs


def _ref_attention(q, k, v, b_start_loc, b_seq_len, is_causal, sm_scale):
    """Reference attention over the packed [tokens, heads, dim] layout."""
    out = torch.zeros_like(q)
    for b in range(b_seq_len.shape[0]):
        start = int(b_start_loc[b].item())
        seq = int(b_seq_len[b].item())
        qb = q[start : start + seq].float()
        kb = k[start : start + seq].float()
        vb = v[start : start + seq].float()
        # GQA: repeat KV heads to match Q heads.
        kv_group = q.shape[1] // k.shape[1]
        kb = kb.repeat_interleave(kv_group, dim=1)
        vb = vb.repeat_interleave(kv_group, dim=1)
        scores = torch.einsum("thd,shd->hts", qb, kb) * sm_scale
        if is_causal:
            mask = torch.triu(
                torch.ones(seq, seq, device=q.device, dtype=torch.bool), 1
            )
            scores = scores.masked_fill(mask[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[start : start + seq] = torch.einsum("hts,shd->thd", probs, vb).to(
            q.dtype
        )
    return out


class TestPrefillAttentionSplitD(CustomTestCase):
    def _run(self, head_dim, dtype=torch.bfloat16, seq=128, heads=8, kv_heads=2):
        torch.manual_seed(0)
        device = "cuda"
        batch = 2
        total = batch * seq
        q = torch.randn(total, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(total, kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(total, kv_heads, head_dim, device=device, dtype=dtype)
        o = torch.empty_like(q)
        b_start_loc = torch.arange(0, total, seq, device=device, dtype=torch.int32)
        b_seq_len = torch.full((batch,), seq, device=device, dtype=torch.int32)
        sm_scale = 1.0 / (head_dim**0.5)
        context_attention_fwd(
            q, k, v, o, b_start_loc, b_seq_len, seq, is_causal=True, sm_scale=sm_scale
        )
        ref = _ref_attention(q, k, v, b_start_loc, b_seq_len, True, sm_scale)
        return o, ref

    @_DEVICE_SKIP
    def test_split_d_matches_reference(self):
        # 72 -> 64 + 16 exercises the tail block; 128 stays single-block.
        for head_dim in (72, 80, 96, 128):
            with self.subTest(head_dim=head_dim):
                out, ref = self._run(head_dim)
                torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    @_DEVICE_SKIP
    def test_split_d_tail_offset_matches_reference(self):
        # The tail block starts at BLOCK_DMODEL, not 0: 72 -> 64 + 16 and
        # 96 -> 64 + 32 both read a non-zero tail offset.
        for head_dim in (72, 96):
            with self.subTest(head_dim=head_dim):
                with mock.patch.object(
                    prefill_ops, "_on_gfx1151", return_value=True
                ):
                    out, ref = self._run(head_dim)
                torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    @_DEVICE_SKIP
    def test_non_gfx1151_path_matches_reference(self):
        # With the guard off the kernel must take the stock single-block path
        # and still match the reference elementwise.
        for head_dim in (72, 128):
            with self.subTest(head_dim=head_dim):
                with mock.patch.object(
                    prefill_ops, "_on_gfx1151", return_value=False
                ):
                    out, ref = self._run(head_dim)
                torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    @_DEVICE_SKIP
    def test_non_contiguous_head_stride_falls_back(self):
        # A padded head axis gives a stride the alignment hint cannot claim.
        torch.manual_seed(1)
        device = "cuda"
        head_dim, pad = 72, 8
        total, heads, kv_heads = 256, 8, 2
        q_wide = torch.randn(total, heads, head_dim + pad, device=device, dtype=torch.bfloat16)
        k_wide = torch.randn(total, kv_heads, head_dim + pad, device=device, dtype=torch.bfloat16)
        v_wide = torch.randn(total, kv_heads, head_dim + pad, device=device, dtype=torch.bfloat16)
        q, k, v = q_wide[..., :head_dim], k_wide[..., :head_dim], v_wide[..., :head_dim]
        o = torch.empty_like(q)
        b_start_loc = torch.tensor([0, 128], device=device, dtype=torch.int32)
        b_seq_len = torch.tensor([128, 128], device=device, dtype=torch.int32)
        sm_scale = 1.0 / (head_dim**0.5)
        context_attention_fwd(
            q, k, v, o, b_start_loc, b_seq_len, 128, is_causal=True, sm_scale=sm_scale
        )
        ref = _ref_attention(q, k, v, b_start_loc, b_seq_len, True, sm_scale)
        torch.testing.assert_close(o, ref, rtol=2e-2, atol=2e-2)

    def _capture_launch(
        self,
        *,
        on_gfx1151,
        head_dim=72,
        head_stride_pad=0,
        dtype=torch.bfloat16,
        max_input_len=64,
    ):
        capture = _LaunchCapture()
        original = prefill_ops._fwd_kernel
        prefill_ops._fwd_kernel = capture
        try:
            with mock.patch.object(
                prefill_ops, "_on_gfx1151", return_value=on_gfx1151
            ):
                def meta(tokens, heads, dim):
                    wide = torch.empty(
                        (tokens, heads, dim + head_stride_pad),
                        dtype=dtype,
                        device="meta",
                    )
                    return wide[..., :dim]

                seq_lens = torch.empty(2, dtype=torch.int32, device="meta")
                context_attention_fwd(
                    meta(64, 8, head_dim),
                    meta(64, 2, head_dim),
                    meta(64, 2, head_dim),
                    meta(64, 8, head_dim),
                    seq_lens,
                    seq_lens,
                    max_input_len,
                )
        finally:
            prefill_ops._fwd_kernel = original
        return capture

    def test_launch_config_guarded(self):
        # Without the gfx1151 guard the launch must be the stock single block.
        capture = self._capture_launch(on_gfx1151=False)
        self.assertEqual(capture.kwargs["BLOCK_DMODEL"], 128)
        self.assertEqual(capture.kwargs["BLOCK_DMODEL_TAIL"], 0)
        self.assertFalse(capture.kwargs["HEAD_STRIDE_ALIGNED_8"])
        self.assertEqual(capture.kwargs["BLOCK_N"], 128)

    def test_launch_config_split_d_on_gfx1151(self):
        # With the guard on and aligned strides, 72 -> 64 + 16 and BLOCK_N=16.
        capture = self._capture_launch(on_gfx1151=True)
        self.assertEqual(capture.kwargs["BLOCK_DMODEL"], 64)
        self.assertEqual(capture.kwargs["BLOCK_DMODEL_TAIL"], 16)
        self.assertTrue(capture.kwargs["HEAD_STRIDE_ALIGNED_8"])
        self.assertEqual(capture.kwargs["BLOCK_N"], 16)
        self.assertEqual(capture.kwargs["num_warps"], 4)

    def test_launch_config_unaligned_stride_falls_back(self):
        # A padded head axis must disable the hint and the narrow tile.
        capture = self._capture_launch(on_gfx1151=True, head_stride_pad=8)
        self.assertEqual(capture.kwargs["BLOCK_DMODEL"], 64)
        self.assertEqual(capture.kwargs["BLOCK_DMODEL_TAIL"], 16)
        self.assertFalse(capture.kwargs["HEAD_STRIDE_ALIGNED_8"])
        self.assertEqual(capture.kwargs["BLOCK_N"], 128)

    def test_launch_config_power_of_two_head_dim_unchanged(self):
        # 128 is already a power of two: no split, no narrow tile, even on
        # gfx1151 with aligned strides.
        capture = self._capture_launch(on_gfx1151=True, head_dim=128)
        self.assertEqual(capture.kwargs["BLOCK_DMODEL"], 128)
        self.assertEqual(capture.kwargs["BLOCK_DMODEL_TAIL"], 0)
        self.assertTrue(capture.kwargs["HEAD_STRIDE_ALIGNED_8"])
        self.assertEqual(capture.kwargs["BLOCK_N"], 128)

    def test_launch_config_skips_non_narrowing_split(self):
        # 112 would be 64 + 64 = 128 lanes, the same extent as next_pow2(112),
        # so the split must be skipped and the stock tile kept.
        capture = self._capture_launch(on_gfx1151=True, head_dim=112)
        self.assertEqual(capture.kwargs["BLOCK_DMODEL"], 128)
        self.assertEqual(capture.kwargs["BLOCK_DMODEL_TAIL"], 0)
        self.assertEqual(capture.kwargs["BLOCK_N"], 128)

    def test_launch_config_fp32_keeps_stock_path(self):
        # The split-D path is gated on bf16/fp16; fp32 must stay single-block.
        capture = self._capture_launch(
            on_gfx1151=True, head_dim=72, dtype=torch.float32
        )
        self.assertEqual(capture.kwargs["BLOCK_DMODEL"], 128)
        self.assertEqual(capture.kwargs["BLOCK_DMODEL_TAIL"], 0)
        self.assertFalse(capture.kwargs["HEAD_STRIDE_ALIGNED_8"])
        self.assertEqual(capture.kwargs["BLOCK_N"], 128)

    def test_launch_config_grid_covers_max_seq_len(self):
        # The grid must still be cdiv(max_input_len, BLOCK_M) for every
        # batch/seq_len combination, independent of the split-D tile.
        for max_input_len in (1, 63, 64, 65, 128, 1024):
            with self.subTest(max_input_len=max_input_len):
                capture = self._capture_launch(
                    on_gfx1151=True, head_dim=72, max_input_len=max_input_len
                )
                self.assertEqual(
                    capture.grid[2], (max_input_len + 127) // 128
                )


if __name__ == "__main__":
    unittest.main()
