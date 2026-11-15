import unittest

import pytest

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import parametrize, precision
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=54, suite="stage-a-test-cpu-intel")

torch.manual_seed(1234)


def _has_flash_attn_varlen_func():
    """Return True only when the flash_attn_varlen_func op is registered.

    The empty-page-table regression tests below exercise the paged KV launch
    guard. On CPU-only CI images the op may not be registered, in which case
    the tests must be skipped rather than failing during collection.
    """
    try:
        return hasattr(torch.ops.sgl_kernel, "flash_attn_varlen_func")
    except Exception:
        return False


_HAS_FLASH_ATTN_VARLEN_FUNC = _has_flash_attn_varlen_func()

# The paged KV launch guard lives in the CUDA flash_api.cpp; the regression
# tests below can only run when the CUDA op is registered. Skip (rather than
# fail) on CPU-only images so collection never depends on CUDA availability.
if _HAS_FLASH_ATTN_VARLEN_FUNC:
    flash_attn_varlen_func = torch.ops.sgl_kernel.flash_attn_varlen_func
else:
    flash_attn_varlen_func = None

pytestmark = pytest.mark.skipif(
    not _HAS_FLASH_ATTN_VARLEN_FUNC,
    reason="flash_attn_varlen_func op is not registered in this build",
)


def flash_attn_varlen_ref(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    is_causal,
    enable_gqa,
):
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    batch = len(cu_k) - 1

    # [T, H, D] -> [1, H, T, D]
    q, k, v = [x.unsqueeze(0).transpose(1, 2) for x in [q, k, v]]

    B, H, T, D = q.shape
    out = torch.empty(B, H, T, v.size(-1), dtype=q.dtype)
    for b in range(batch):
        start_q, end_q = cu_q[b], cu_q[b + 1]
        start_k, end_k = cu_k[b], cu_k[b + 1]

        out[:, :, start_q:end_q, :] = F.scaled_dot_product_attention(
            q[:, :, start_q:end_q, :],
            k[:, :, start_k:end_k, :],
            v[:, :, start_k:end_k, :],
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )

    # [1, H, T, D] -> [T, H, D]
    return out.transpose(1, 2).squeeze(0)


# faster version ref kernel for non varlen case
def flash_attn_non_varlen_ref(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    is_causal,
    enable_gqa,
):
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    batch = len(cu_k) - 1

    B_T, H, D = q.shape
    T = B_T // batch

    # [T, H, D] -> [1, H, T, D]
    q, k, v = [x.reshape(batch, T, H, D).transpose(1, 2) for x in [q, k, v]]

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )
    # [B, H, T, D] -> [B * T, H, D]
    return out.transpose(1, 2).reshape(batch * T, H, D)


class TestFlashAttn(CustomTestCase):
    @parametrize(
        batch=[4],
        max_seqlen_q=[35, 96],
        max_seqlen_k=[35, 96],
        num_heads=[16],
        num_heads_kv=[16, 2],
        head_dim=[32, 48],  # test when D is not 32x
        head_dim_v=[32],
        is_causal=[True, False],
    )
    def test_flash_attn_varlen(
        self,
        batch,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_kv,
        head_dim,
        head_dim_v,
        is_causal,
    ):
        dtype = torch.bfloat16

        # random seqlens for k and kv
        seqlens_q = torch.randint(1, max_seqlen_q, (batch,), dtype=torch.int32)
        seqlens_k = torch.randint(1, max_seqlen_k, (batch,), dtype=torch.int32)
        cu_seqlens_q = torch.zeros((batch + 1,), dtype=torch.int32)
        cu_seqlens_k = torch.zeros((batch + 1,), dtype=torch.int32)
        cu_seqlens_q[1:] = torch.cumsum(seqlens_q, 0)
        cu_seqlens_k[1:] = torch.cumsum(seqlens_k, 0)

        sum_seqlen_q = seqlens_q.sum().item()
        sum_seqlen_k = seqlens_k.sum().item()
        q = torch.randn(sum_seqlen_q, num_heads, head_dim).to(dtype)
        k = torch.randn(sum_seqlen_k, num_heads_kv, head_dim).to(dtype)
        v = torch.randn(sum_seqlen_k, num_heads_kv, head_dim_v).to(dtype)

        out_ref = flash_attn_varlen_ref(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            is_causal=is_causal,
            enable_gqa=num_heads != num_heads_kv,
        )

        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlens_q.max().item(),
            seqlens_k.max().item(),
            is_causal,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(out_ref, out, atol=atol, rtol=rtol)

    # test with large size to capture overflow issue
    @parametrize(
        batch=[4097],
        max_seqlen_q=[4097],
        max_seqlen_k=[4097],
        num_heads=[4],
        num_heads_kv=[4],
        head_dim=[32],
        head_dim_v=[32],
        is_causal=[False],
    )
    def test_flash_attn_large_size(
        self,
        batch,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_kv,
        head_dim,
        head_dim_v,
        is_causal,
    ):
        dtype = torch.bfloat16

        # test the non varlen case
        seqlens_q = torch.full((batch,), max_seqlen_q, dtype=torch.int32)
        seqlens_k = torch.full((batch,), max_seqlen_k, dtype=torch.int32)

        cu_seqlens_q = torch.zeros((batch + 1,), dtype=torch.int32)
        cu_seqlens_k = torch.zeros((batch + 1,), dtype=torch.int32)
        cu_seqlens_q[1:] = torch.cumsum(seqlens_q, 0)
        cu_seqlens_k[1:] = torch.cumsum(seqlens_k, 0)

        sum_seqlen_q = seqlens_q.sum().item()
        sum_seqlen_k = seqlens_k.sum().item()
        q = torch.randn(sum_seqlen_q, num_heads, head_dim).to(dtype)
        k = torch.randn(sum_seqlen_k, num_heads_kv, head_dim).to(dtype)
        v = torch.randn(sum_seqlen_k, num_heads_kv, head_dim_v).to(dtype)

        out_ref = flash_attn_non_varlen_ref(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            is_causal=is_causal,
            enable_gqa=num_heads != num_heads_kv,
        )

        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlens_q.max().item(),
            seqlens_k.max().item(),
            is_causal,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(out_ref, out, atol=atol, rtol=rtol)

    def _test_flash_attn_large_seq_causal_mask_once(self, seqlens):
        dtype = torch.bfloat16
        num_heads = 8
        num_heads_kv = 2
        head_dim = 64

        seqlens_t = torch.tensor(seqlens, dtype=torch.int32)
        cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(seqlens_t, 0)
        total = cu_seqlens[-1].item()
        max_seqlen = seqlens_t.max().item()

        q = torch.randn(total, num_heads, head_dim, dtype=dtype)
        k = torch.randn(total, num_heads_kv, head_dim, dtype=dtype)
        v = torch.randn(total, num_heads_kv, head_dim, dtype=dtype)

        out_ref = flash_attn_varlen_ref(
            q, k, v, cu_seqlens, cu_seqlens, is_causal=True, enable_gqa=True
        )
        out = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, True
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(out_ref, out, atol=atol, rtol=rtol)

    def test_flash_attn_large_seq_causal_mask(self):
        # Non-varlen path: single sequence, has_varlen_sequences returns False
        # → dispatches to flash_attn_kernel_impl.
        self._test_flash_attn_large_seq_causal_mask_once([5000])
        # Varlen path: sequences with different lengths, has_varlen_sequences
        # returns True → dispatches to flash_attn_varlen_kernel_impl
        self._test_flash_attn_large_seq_causal_mask_once([5000, 4999])

    @unittest.skipUnless(
        _HAS_FLASH_ATTN_VARLEN_FUNC,
        "flash_attn_varlen_func op is not registered in this build",
    )
    @parametrize(
        batch=[2],
        max_seqlen_q=[16],
        num_heads=[4],
        num_heads_kv=[4],
        head_dim=[32],
        head_dim_v=[32],
        is_causal=[True, False],
    )
    def test_flash_attn_empty_page_table_guard(
        self,
        batch,
        max_seqlen_q,
        num_heads,
        num_heads_kv,
        head_dim,
        head_dim_v,
        is_causal,
    ):
        """Regression: an empty page table must not launch the paged KV kernel.

        Mirrors upstream PR #3161. When the page table has zero columns the
        logical key length is zero even though the allocated cache tensor is
        non-empty; the launch guard must short-circuit to the host-side empty
        path instead of reading out-of-bounds page-table entries.
        """
        dtype = torch.bfloat16
        seqlens_q = torch.full((batch,), max_seqlen_q, dtype=torch.int32)
        cu_seqlens_q = torch.zeros((batch + 1,), dtype=torch.int32)
        cu_seqlens_q[1:] = torch.cumsum(seqlens_q, 0)
        total_q = seqlens_q.sum().item()

        q = torch.randn(total_q, num_heads, head_dim, dtype=dtype)
        # Allocated cache with non-empty storage but an empty page table.
        k_cache = torch.randn(4, 256, num_heads_kv, head_dim, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        page_table = torch.empty(batch, 0, dtype=torch.int32)

        # The guard must reject the launch; the call should either return an
        # empty/zero result or raise a clear error, never read out of bounds.
        try:
            out = flash_attn_varlen_func(
                q,
                k_cache,
                v_cache,
                cu_seqlens_q,
                cu_seqlens_q,
                max_seqlen_q,
                0,
                is_causal,
                block_table=page_table,
            )
        except (RuntimeError, ValueError):
            # A clear error is an acceptable outcome for an empty page table.
            self.assertEqual(page_table.shape[1], 0)
            return
        self.assertEqual(out.shape[0], total_q)
        # Empty KV must not produce NaN/Inf garbage in the output.
        self.assertTrue(torch.isfinite(out).all().item())
        # The empty-KV host path returns zero output; assert it explicitly so a
        # regression that launches the paged kernel is caught even if the
        # out-of-bounds read happens to yield finite garbage.
        self.assertTrue((out == 0).all().item())
        # The guard must have short-circuited before the paged kernel: an empty
        # page table has zero logical keys, so the host-side empty path is used.
        self.assertEqual(page_table.shape[1], 0)

    @unittest.skipUnless(
        _HAS_FLASH_ATTN_VARLEN_FUNC,
        "flash_attn_varlen_func op is not registered in this build",
    )
    @parametrize(
        batch=[2],
        max_seqlen_q=[16],
        num_heads=[4],
        num_heads_kv=[4],
        head_dim=[32],
        head_dim_v=[32],
        is_causal=[True, False],
    )
    def test_flash_attn_empty_page_table_guard_placeholder(
        self,
        batch,
        max_seqlen_q,
        num_heads,
        num_heads_kv,
        head_dim,
        head_dim_v,
        is_causal,
    ):
        """Regression: an empty page table must not launch the paged KV kernel.

        Mirrors upstream PR #3161. When the page table has zero columns the
        logical key length is zero even though the allocated cache tensor is
        non-empty; the launch guard must short-circuit to the host-side empty
        path instead of reading out-of-bounds page-table entries.
        """
        dtype = torch.bfloat16
        seqlens_q = torch.full((batch,), max_seqlen_q, dtype=torch.int32)
        cu_seqlens_q = torch.zeros((batch + 1,), dtype=torch.int32)
        cu_seqlens_q[1:] = torch.cumsum(seqlens_q, 0)
        total_q = seqlens_q.sum().item()

        q = torch.randn(total_q, num_heads, head_dim, dtype=dtype)
        # Allocated cache with non-empty storage but an empty page table.
        k_cache = torch.randn(4, 256, num_heads_kv, head_dim, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        page_table = torch.empty(batch, 0, dtype=torch.int32)

        # The guard must reject the launch; the call should either return an
        # empty/zero result or raise a clear error, never read out of bounds.
        try:
            out = flash_attn_varlen_func(
                q,
                k_cache,
                v_cache,
                cu_seqlens_q,
                cu_seqlens_q,
                max_seqlen_q,
                0,
                is_causal,
                block_table=page_table,
            )
        except (RuntimeError, ValueError):
            # A clear error is an acceptable outcome for an empty page table.
            self.assertEqual(page_table.shape[1], 0)
            return
        self.assertEqual(out.shape[0], total_q)
        # Empty KV must not produce NaN/Inf garbage in the output.
        self.assertTrue(torch.isfinite(out).all().item())
        # The empty-KV host path returns zero output; assert it explicitly so a
        # regression that launches the paged kernel is caught even if the
        # out-of-bounds read happens to yield finite garbage.
        self.assertTrue((out == 0).all().item())
        # The guard must have short-circuited before the paged kernel: an empty
        # page table has zero logical keys, so the host-side empty path is used.
        self.assertEqual(page_table.shape[1], 0)

    @unittest.skipUnless(
        _HAS_FLASH_ATTN_VARLEN_FUNC,
        "flash_attn_varlen_func op is not registered in this build",
    )
    @parametrize(
        batch=[2],
        max_seqlen_q=[16],
        num_heads=[4],
        num_heads_kv=[4],
        head_dim=[32],
        head_dim_v=[32],
        is_causal=[True, False],
    )
    def test_flash_attn_empty_page_table_view_guard(
        self,
        batch,
        max_seqlen_q,
        num_heads,
        num_heads_kv,
        head_dim,
        head_dim_v,
        is_causal,
    ):
        """Regression: an empty page-table view with backing storage must not launch.

        Mirrors upstream PR #3161's `page_table_view` case: the page table is a
        zero-column slice of a non-empty tensor, so the backing storage exists
        but there are no logical keys. The launch guard must short-circuit to
        the host-side empty path instead of reading out-of-bounds entries.
        """
        dtype = torch.bfloat16
        seqlens_q = torch.full((batch,), max_seqlen_q, dtype=torch.int32)
        cu_seqlens_q = torch.zeros((batch + 1,), dtype=torch.int32)
        cu_seqlens_q[1:] = torch.cumsum(seqlens_q, 0)
        total_q = seqlens_q.sum().item()

        q = torch.randn(total_q, num_heads, head_dim, dtype=dtype)
        k_cache = torch.randn(4, 256, num_heads_kv, head_dim, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        # Zero-column view over a non-empty backing tensor.
        page_table = torch.zeros(batch, 4, dtype=torch.int32)[:, :0]

        try:
            out = flash_attn_varlen_func(
                q,
                k_cache,
                v_cache,
                cu_seqlens_q,
                cu_seqlens_q,
                max_seqlen_q,
                0,
                is_causal,
                block_table=page_table,
            )
        except (RuntimeError, ValueError):
            # A clear error is an acceptable outcome for an empty page table.
            self.assertEqual(page_table.shape[1], 0)
            return
        self.assertEqual(out.shape[0], total_q)
        # Empty KV must not produce NaN/Inf garbage in the output.
        self.assertTrue(torch.isfinite(out).all().item())
        # The empty-KV host path returns zero output; assert it explicitly so a
        # regression that launches the paged kernel is caught even if the
        # out-of-bounds read happens to yield finite garbage.
        self.assertTrue((out == 0).all().item())
        # The zero-column view must still be recognized as an empty page table.
        self.assertEqual(page_table.shape[1], 0)
        # The view keeps backing storage but has no logical page-table entries.
        self.assertGreater(page_table.storage().size(), 0)

    @unittest.skipUnless(
        _HAS_FLASH_ATTN_VARLEN_FUNC,
        "flash_attn_varlen_func op is not registered in this build",
    )
    @parametrize(
        batch=[2],
        max_seqlen_q=[16],
        num_heads=[4],
        num_heads_kv=[4],
        head_dim=[32],
        head_dim_v=[32],
        is_causal=[True, False],
    )
    def test_flash_attn_nonempty_page_table_guard(
        self,
        batch,
        max_seqlen_q,
        num_heads,
        num_heads_kv,
        head_dim,
        head_dim_v,
        is_causal,
    ):
        """The added guard must not reject legitimate non-empty page tables."""
        dtype = torch.bfloat16
        seqlens_q = torch.full((batch,), max_seqlen_q, dtype=torch.int32)
        cu_seqlens_q = torch.zeros((batch + 1,), dtype=torch.int32)
        cu_seqlens_q[1:] = torch.cumsum(seqlens_q, 0)
        total_q = seqlens_q.sum().item()

        q = torch.randn(total_q, num_heads, head_dim, dtype=dtype)
        k_cache = torch.randn(4, 256, num_heads_kv, head_dim, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        page_table = torch.zeros(batch, 1, dtype=torch.int32)

        out = flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            cu_seqlens_q,
            max_seqlen_q,
            256,
            is_causal,
            block_table=page_table,
        )
        self.assertEqual(out.shape[0], total_q)
        # A non-empty page table must still take the paged kernel path and
        # produce finite output; the guard must not reject legitimate input.
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertEqual(page_table.shape[1], 1)


if __name__ == "__main__":
    unittest.main()
