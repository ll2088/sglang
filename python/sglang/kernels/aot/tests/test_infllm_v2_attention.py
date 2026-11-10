"""Equivalence tests for the migrated InfLLM-V2 FlashAttention API.

These compare the ``sgl_kernel.infllm_v2`` implementations against the original
``infllm_v2`` package (3rdparty/infllmv2_cuda_impl). Both call the same CUDA
kernels, so outputs are expected to match closely. The whole module is skipped
if the reference ``infllm_v2`` package is not importable.
"""

import pytest
import torch

sgl = pytest.importorskip("sgl_kernel.infllm_v2")
ref = pytest.importorskip("infllm_v2")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for InfLLM-V2 kernels"
)


def _assert_close(a, b, name):
    a = a.float()
    b = b.float()
    assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} vs {b.shape}"
    max_diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, atol=1e-2, rtol=1e-2), f"{name}: max diff {max_diff}"


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(256, 16), (64, 17)])
def test_stage1_matches_reference(head_dim, causal, seqlen_q, seqlen_k):
    torch.manual_seed(0)
    n_heads, n_kv_heads = 32, 2
    dtype = torch.bfloat16

    q = torch.randn(n_heads, seqlen_q, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(n_kv_heads, seqlen_k, head_dim, dtype=dtype, device="cuda")

    cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device="cuda")

    q = q.transpose(0, 1).contiguous()
    k = k.transpose(0, 1).contiguous()

    common = dict(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cu_seqlens_v=cu_seqlens_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        causal=causal,
    )
    out_ref = ref.infllmv2_attn_stage1(q, k, k, **common)
    out_sgl = sgl.infllmv2_attn_stage1(q, k, k, **common)
    _assert_close(out_sgl, out_ref, "stage1")


def test_stage1_seqused_k_length_must_equal_batch_size():
    torch.manual_seed(0)
    n_heads, n_kv_heads, seqlen_q, seqlen_k, head_dim = 8, 2, 16, 16, 64
    dtype = torch.bfloat16

    q = torch.randn(seqlen_q, n_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(seqlen_k, n_kv_heads, head_dim, dtype=dtype, device="cuda")
    cu_seqlens = torch.tensor([0, seqlen_q], dtype=torch.int32, device="cuda")

    common = dict(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        cu_seqlens_v=cu_seqlens,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        causal=False,
    )

    # Positive control: one index per sequence must NOT raise.
    out = sgl.infllmv2_attn_stage1(
        q, k, k, seqused_k=torch.tensor([seqlen_k], dtype=torch.int32, device="cuda"), **common
    )
    assert out.shape == (seqlen_q, n_heads, head_dim)
    assert not out.isnan().any()

    # Regression: a seqused_k whose length is not batch_size must be rejected.
    # Slices of one buffer, as in the upstream issue: the memory right after a
    # short slice holds valid indices, so only the explicit length check catches it.
    buf = torch.tensor([seqlen_k, seqlen_k, seqlen_k], dtype=torch.int32, device="cuda")
    for seqused_k in (buf[:0], buf[:2], buf):  # 0, 2 and 3 indices for 1 sequence
        with pytest.raises(RuntimeError, match="seqused_k must have shape"):
            sgl.infllmv2_attn_stage1(q, k, k, seqused_k=seqused_k, **common)


def test_stage1_leftpad_k_length_must_equal_batch_size():
    torch.manual_seed(0)
    n_heads, n_kv_heads, seqlen_q, seqlen_k, head_dim = 8, 2, 16, 16, 64
    dtype = torch.bfloat16

    q = torch.randn(seqlen_q, n_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(seqlen_k, n_kv_heads, head_dim, dtype=dtype, device="cuda")
    cu_seqlens = torch.tensor([0, seqlen_q], dtype=torch.int32, device="cuda")

    common = dict(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        cu_seqlens_v=cu_seqlens,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        causal=False,
    )

    # Positive control: one index per sequence must NOT raise.
    out = sgl.infllmv2_attn_stage1(
        q, k, k, leftpad_k=torch.tensor([0], dtype=torch.int32, device="cuda"), **common
    )
    assert out.shape == (seqlen_q, n_heads, head_dim)
    assert not out.isnan().any()

    # Regression: a leftpad_k whose length is not batch_size must be rejected.
    buf = torch.tensor([0, 0, 0], dtype=torch.int32, device="cuda")
    for leftpad_k in (buf[:0], buf[:2], buf):  # 0, 2 and 3 indices for 1 sequence
        with pytest.raises(RuntimeError, match="leftpad_k must have shape"):
            sgl.infllmv2_attn_stage1(q, k, k, leftpad_k=leftpad_k, **common)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
