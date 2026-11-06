import unittest

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import parametrize, precision
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=54, suite="stage-a-test-cpu-intel")

# Importing sgl_kernel may initialize CUDA; keep module import side-effect free so
# that pytest collection works on CPU-only CI runners.
try:
    import sgl_kernel  # noqa: F401

    flash_attn_varlen_func = torch.ops.sgl_kernel.flash_attn_varlen_func
    _HAS_SGL_KERNEL = True
except Exception:  # pragma: no cover - depends on build environment
    flash_attn_varlen_func = None
    _HAS_SGL_KERNEL = False

# Avoid initializing CUDA at import time on CPU-only CI runners; the kernel
# tests are skipped unless both sgl_kernel and CUDA are available.
_HAS_CUDA = _HAS_SGL_KERNEL and torch.cuda.is_available()


def _requires_kernel(test_func):
    """Skip tests that need the compiled sgl_kernel / CUDA at runtime."""
    return unittest.skipUnless(
        _HAS_SGL_KERNEL and _HAS_CUDA,
        "requires sgl_kernel with CUDA support",
    )(test_func)

torch.manual_seed(1234)


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
    @_requires_kernel
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
    @_requires_kernel
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

    @_requires_kernel
    def test_flash_attn_large_seq_causal_mask(self):
        # Non-varlen path: single sequence, has_varlen_sequences returns False
        # → dispatches to flash_attn_kernel_impl.
        self._test_flash_attn_large_seq_causal_mask_once([5000])
        # Varlen path: sequences with different lengths, has_varlen_sequences
        # returns True → dispatches to flash_attn_varlen_kernel_impl
        self._test_flash_attn_large_seq_causal_mask_once([5000, 4999])

    @_requires_kernel
    @parametrize(
        batch=[3],
        seqlen_q=[128, 256],
        seqlen_k=[256, 512],
        num_heads=[2],
        num_heads_kv=[2],
        head_dim=[64, 128],
        is_causal=[True, False],
    )
    def test_flash_attn_leftpad_no_seqlens(
        self,
        batch,
        seqlen_q,
        seqlen_k,
        num_heads,
        num_heads_kv,
        head_dim,
        is_causal,
    ):
        # Regression test for the even-MN fast path: when only leftpad is given
        # (no per-sequence seqlens), seqlen_k - leftpad[b] need not be block aligned,
        # so the kernel must not skip masking the last key block.
        dtype = torch.bfloat16
        leftpad = torch.tensor([31, 0, 17][:batch], dtype=torch.int32)

        q = torch.randn(batch, seqlen_q, num_heads, head_dim, dtype=dtype)
        k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, dtype=dtype)
        v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, dtype=dtype)

        # Reference: per-sequence attention over the valid (non-leftpad) key range.
        out_ref = torch.empty(batch, seqlen_q, num_heads, head_dim, dtype=dtype)
        for b in range(batch):
            pad = int(leftpad[b].item())
            k_b = k[b, pad:]
            v_b = v[b, pad:]
            q_b = q[b].transpose(0, 1).unsqueeze(0)  # [1, H, Sq, D]
            k_b = k_b.transpose(0, 1).unsqueeze(0)  # [1, H, Sk, D]
            v_b = v_b.transpose(0, 1).unsqueeze(0)
            out_b = F.scaled_dot_product_attention(
                q_b, k_b, v_b, is_causal=is_causal, enable_gqa=num_heads != num_heads_kv
            )
            out_ref[b] = out_b.squeeze(0).transpose(0, 1)

        # Flatten to varlen layout with cu_seqlens_k encoding the leftpad offset,
        # which exercises the same BlockInfo actual_seqlen_k path as leftpad_k.
        cu_seqlens_q = torch.arange(0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32)
        for b in range(batch):
            cu_seqlens_k[b + 1] = cu_seqlens_k[b] + seqlen_k - int(leftpad[b].item())

        q_flat = q.reshape(batch * seqlen_q, num_heads, head_dim)
        k_flat = torch.cat([k[b, int(leftpad[b].item()):] for b in range(batch)], dim=0)
        v_flat = torch.cat([v[b, int(leftpad[b].item()):] for b in range(batch)], dim=0)

        out = flash_attn_varlen_func(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            is_causal,
        )
        out = out.reshape(batch, seqlen_q, num_heads, head_dim)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(out_ref, out, atol=atol, rtol=rtol)

    @_requires_kernel
    @parametrize(
        batch=[3],
        seqlen_q=[128, 256],
        seqlen_k=[256, 512],
        num_heads=[2],
        num_heads_kv=[2],
        head_dim=[64, 128],
        is_causal=[True, False],
    )
    def test_flash_attn_leftpad_with_seqlens_consistency(
        self,
        batch,
        seqlen_q,
        seqlen_k,
        num_heads,
        num_heads_kv,
        head_dim,
        is_causal,
    ):
        # When both leftpad and seqlens are provided, the result must match the
        # seqlens-only path (leftpad is already encoded in cu_seqlens_k).
        dtype = torch.bfloat16
        leftpad = torch.tensor([31, 0, 17][:batch], dtype=torch.int32)
        seqlens_k = torch.tensor([seqlen_k - int(p) for p in leftpad], dtype=torch.int32)

        q = torch.randn(batch, seqlen_q, num_heads, head_dim, dtype=dtype)
        k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, dtype=dtype)
        v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, dtype=dtype)

        cu_seqlens_q = torch.arange(0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32)
        cu_seqlens_k[1:] = torch.cumsum(seqlens_k, 0)

        q_flat = q.reshape(batch * seqlen_q, num_heads, head_dim)
        k_flat = torch.cat([k[b, int(leftpad[b].item()):] for b in range(batch)], dim=0)
        v_flat = torch.cat([v[b, int(leftpad[b].item()):] for b in range(batch)], dim=0)

        out = flash_attn_varlen_func(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            is_causal,
        )
        out = out.reshape(batch, seqlen_q, num_heads, head_dim)

        # Reference computed with the same varlen layout but without leftpad.
        out_ref = flash_attn_varlen_ref(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens_q,
            cu_seqlens_k,
            is_causal=is_causal,
            enable_gqa=num_heads != num_heads_kv,
        )
        out_ref = out_ref.reshape(batch, seqlen_q, num_heads, head_dim)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(out_ref, out, atol=atol, rtol=rtol)

    @_requires_kernel
    @parametrize(
        batch=[2],
        seqlen_q=[127, 255],
        seqlen_k=[255, 511],
        num_heads=[2],
        num_heads_kv=[2],
        head_dim=[64],
        is_causal=[True, False],
    )
    def test_flash_attn_leftpad_no_seqlens_odd_shape(
        self,
        batch,
        seqlen_q,
        seqlen_k,
        num_heads,
        num_heads_kv,
        head_dim,
        is_causal,
    ):
        # Boundary shapes where seqlen is not a power of two and not block aligned.
        dtype = torch.bfloat16
        leftpad = torch.tensor([31, 0][:batch], dtype=torch.int32)

        q = torch.randn(batch, seqlen_q, num_heads, head_dim, dtype=dtype)
        k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, dtype=dtype)
        v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, dtype=dtype)

        out_ref = torch.empty(batch, seqlen_q, num_heads, head_dim, dtype=dtype)
        for b in range(batch):
            pad = int(leftpad[b].item())
            q_b = q[b].transpose(0, 1).unsqueeze(0)
            k_b = k[b, pad:].transpose(0, 1).unsqueeze(0)
            v_b = v[b, pad:].transpose(0, 1).unsqueeze(0)
            out_b = F.scaled_dot_product_attention(
                q_b, k_b, v_b, is_causal=is_causal, enable_gqa=num_heads != num_heads_kv
            )
            out_ref[b] = out_b.squeeze(0).transpose(0, 1)

        cu_seqlens_q = torch.arange(0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32)
        for b in range(batch):
            cu_seqlens_k[b + 1] = cu_seqlens_k[b] + seqlen_k - int(leftpad[b].item())

        q_flat = q.reshape(batch * seqlen_q, num_heads, head_dim)
        k_flat = torch.cat([k[b, int(leftpad[b].item()):] for b in range(batch)], dim=0)
        v_flat = torch.cat([v[b, int(leftpad[b].item()):] for b in range(batch)], dim=0)

        out = flash_attn_varlen_func(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            is_causal,
        )
        out = out.reshape(batch, seqlen_q, num_heads, head_dim)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(out_ref, out, atol=atol, rtol=rtol)

    def test_flash_attn_even_mn_branch_excludes_leftpad(self):
        # Static assertion on the dispatch condition: the even-MN fast path must be
        # disabled whenever leftpad_k or seqused_k is set, even if seqlen is block aligned.
        # This mirrors the C++ condition in flash_fwd_launch_template.h.
        def is_even_mn(cu_seqlens_q, cu_seqlens_k, leftpad_k, seqused_k, seqlen_k, seqlen_q, block_n, block_m):
            return (
                cu_seqlens_q is None
                and cu_seqlens_k is None
                and leftpad_k is None
                and seqused_k is None
                and seqlen_k % block_n == 0
                and seqlen_q % block_m == 0
            )

        # Block-aligned shapes with leftpad must NOT take the even-MN path.
        self.assertFalse(is_even_mn(None, None, object(), None, 256, 128, 128, 64))
        self.assertFalse(is_even_mn(None, None, None, object(), 256, 128, 128, 64))
        # Without leftpad/seqused, block-aligned shapes may take the fast path.
        self.assertTrue(is_even_mn(None, None, None, None, 256, 128, 128, 64))
        # Non-aligned shapes never take the fast path.
        self.assertFalse(is_even_mn(None, None, None, None, 255, 128, 128, 64))


if __name__ == "__main__":
    unittest.main()
