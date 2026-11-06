"""Split-D head-dim blocking and stride-alignment hint for prefill attention.

Covers the gfx1151-only path added to
``sglang.kernels.ops.attention.prefill_attention``: the split-D tail block,
the ``tl.multiple_of`` head-stride hint, and the conditional launch config.
The non-gfx1151 path must stay bit-identical to the single-block kernel.
"""

import unittest

import torch

from sglang.kernels.ops.attention import prefill_attention as prefill_ops
from sglang.kernels.ops.attention.prefill_attention import context_attention_fwd
from sglang.test.test_utils import CustomTestCase


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

    def test_split_d_matches_reference(self):
        # 72 -> 64 + 16 exercises the tail block; 128 stays single-block.
        for head_dim in (72, 80, 96, 128):
            with self.subTest(head_dim=head_dim):
                out, ref = self._run(head_dim)
                torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

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

    def test_launch_config_guarded(self):
        # Without the gfx1151 guard the launch must be the stock single block.
        capture = _LaunchCapture()
        original = prefill_ops._fwd_kernel
        prefill_ops._fwd_kernel = capture
        try:
            q = torch.empty((64, 8, 72), dtype=torch.bfloat16, device="meta")
            k = torch.empty((64, 2, 72), dtype=torch.bfloat16, device="meta")
            v = torch.empty((64, 2, 72), dtype=torch.bfloat16, device="meta")
            o = torch.empty((64, 8, 72), dtype=torch.bfloat16, device="meta")
            seq_lens = torch.empty(2, dtype=torch.int32, device="meta")
            context_attention_fwd(q, k, v, o, seq_lens, seq_lens, 64)
        finally:
            prefill_ops._fwd_kernel = original
        self.assertEqual(capture.kwargs["BLOCK_DMODEL"], 128)
        self.assertEqual(capture.kwargs["BLOCK_DMODEL_TAIL"], 0)
        self.assertFalse(capture.kwargs["HEAD_STRIDE_ALIGNED_8"])


if __name__ == "__main__":
    unittest.main()
