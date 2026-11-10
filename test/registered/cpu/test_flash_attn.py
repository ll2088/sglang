import unittest

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import parametrize, precision
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=54, suite="stage-a-test-cpu-intel")

flash_attn_varlen_func = torch.ops.sgl_kernel.flash_attn_varlen_func

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

    def _skip_if_free_memory_below(self, gib):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        free, _ = torch.cuda.mem_get_info()
        if free < gib * 2**30:
            self.skipTest(f"needs {gib} GiB of free GPU memory")

    def _with_row_stride(self, x, row_stride):
        """Return the values of `x` (batch 1) with consecutive rows `row_stride` elements apart."""
        _, seqlen, nheads, d = x.shape
        buf = torch.empty(
            (seqlen - 1) * row_stride + nheads * d, device=x.device, dtype=x.dtype
        )
        y = buf.as_strided(x.shape, (seqlen * row_stride, row_stride, d, 1))
        y.copy_(x)
        return y

    def test_flash_attn_kvcache_large_row_stride(self):
        # The split-KV kernel steps K/V back one block of kBlockN rows at a time.
        # A row stride of 2**31 / kBlockN or more used to wrap in 32-bit arithmetic.
        self._skip_if_free_memory_below(12)
        torch.random.manual_seed(0)
        device, dtype = "cuda", torch.bfloat16
        nheads, d, row_stride = 2, 256, 2**25 + 128
        seqlen_k = 65
        q = torch.randn(1, 1, nheads, d, device=device, dtype=dtype)
        k_cache = torch.randn(1, seqlen_k, nheads, d, device=device, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        kwargs = dict(
            cache_batch_idx=torch.zeros(1, device=device, dtype=torch.int32),
            num_splits=1,
        )

        def run(strided_name):
            args = {"k_cache": k_cache.clone(), "v_cache": v_cache.clone()}
            if strided_name is not None:
                args[strided_name] = self._with_row_stride(args[strided_name], row_stride)
            out = flash_attn_with_kvcache(
                q, args["k_cache"], args["v_cache"], **kwargs
            )
            return out, args["k_cache"], args["v_cache"]

        out_ref, k_cache_ref, v_cache_ref = run(None)
        out, k_cache_out, v_cache_out = run("k_cache")
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, out_ref))
        self.assertTrue(torch.equal(k_cache_out, k_cache_ref))
        self.assertTrue(torch.equal(v_cache_out, v_cache_ref))

    def test_flash_attn_kvcache_splitkv_large_accum(self):
        # The split-KV out_accum offsets and the combine kernel's offsets were
        # 32-bit products; compare against the unsplit kernel.
        self._skip_if_free_memory_below(16)
        torch.random.manual_seed(0)
        device, dtype = "cuda", torch.bfloat16
        nheads, d, seqlen_q, num_splits = 2, 128, 81920, 16
        q = torch.randn(1, seqlen_q, nheads, d, device=device, dtype=dtype)
        k_cache = torch.randn(1, 128, nheads, d, device=device, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
        kwargs = dict(
            cache_batch_idx=torch.zeros(1, device=device, dtype=torch.int32),
            cache_seqlens=torch.full((1,), 128, device=device, dtype=torch.int32),
        )
        out_ref = flash_attn_with_kvcache(
            q, k_cache, v_cache, num_splits=1, **kwargs
        )
        out = flash_attn_with_kvcache(
            q, k_cache, v_cache, num_splits=num_splits, **kwargs
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
