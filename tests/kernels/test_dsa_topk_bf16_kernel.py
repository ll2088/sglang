"""Kernel-level BF16 top-k tests, guarded so they can be collected in a
minimal CI job without CUDA or the compiled sglang extensions.

When the compiled ops are unavailable every test is reported as skipped; the
pure-Python reference tests live in ``test_dsa_topk_bf16.py`` and always run.
"""

from __future__ import annotations

import unittest


def _ops_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    try:
        import torch  # noqa: F811

        return hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "cooperative_topk")
    except Exception:
        return False


OPS_AVAILABLE = _ops_available()


@unittest.skipUnless(OPS_AVAILABLE, "requires compiled sglang kernels")
class TestCooperativeTopkBf16(unittest.TestCase):
    def test_rows_shorter_than_stride(self) -> None:
        import torch

        torch.manual_seed(0)
        num_rows, stride, top_k = 4, 65536, 512
        row_lengths = [300000, 40000, 16384, 5000]
        lengths = torch.tensor(
            [min(stride, row_lengths[i % len(row_lengths)]) for i in range(num_rows)],
            dtype=torch.int32,
            device="cuda",
        )
        for dtype in (torch.float32, torch.bfloat16):
            logits = torch.randn(num_rows, stride, device="cuda", dtype=dtype)
            out = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")
            workspace = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
            torch.ops._C.cooperative_topk(logits, lengths, out, workspace, top_k)
            for row in range(num_rows):
                length = int(lengths[row])
                k = min(top_k, length)
                idx = out[row, :k].long()
                self.assertEqual(idx.unique().numel(), k)
                torch.testing.assert_close(
                    logits[row, idx].sort().values,
                    logits[row, :length].topk(k).values.sort().values,
                )

    def test_overflow_cluster_boundary_bf16(self) -> None:
        import torch

        num_rows, seq_len, top_k = 9, 65537, 2048
        stride = (seq_len + 7) & ~7
        row = 1.0 + torch.arange(seq_len, dtype=torch.float32) * (0.01 / seq_len)
        logits = torch.full((num_rows, stride), float("inf"), device="cuda", dtype=torch.bfloat16)
        logits[:, :seq_len] = row.cuda().to(torch.bfloat16)
        lengths = torch.full((num_rows,), seq_len, dtype=torch.int32, device="cuda")
        indices = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
        workspace = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
        torch.ops._C.cooperative_topk(logits, lengths, indices, workspace, top_k)
        for r in range(num_rows):
            self.assertEqual(indices[r].unique().numel(), top_k)


if __name__ == "__main__":
    unittest.main()
