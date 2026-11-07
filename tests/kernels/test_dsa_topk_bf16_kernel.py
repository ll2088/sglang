import pytest
import torch


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _has_extension() -> bool:
    try:
        import sglang.kernels.jit.csrc.dsa.kpool_topk_transform  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_has_cuda() and _has_extension()),
    reason="requires CUDA and the compiled DSA top-k extension",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_rows", [1, 4, 8, 33, 64])
@pytest.mark.parametrize("stride", [65536, 1048576])
def test_cooperative_topk_bf16_matches_fp32(dtype, num_rows, stride):
    from sglang.kernels.jit.csrc.dsa.kpool_topk_transform import kpool_topk_transform

    torch.manual_seed(0)
    row_lengths = [min(stride, 300000), 40000, 16384, 5000, 300]
    lengths = [row_lengths[i % len(row_lengths)] for i in range(num_rows)]
    logits = torch.randn(num_rows, stride, device="cuda").to(dtype)
    lengths_t = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    out = torch.empty(num_rows, 2048, dtype=torch.int32, device="cuda")
    kpool_topk_transform(logits, lengths_t, out, 1, None, None, None, None, None)
    for row, length in enumerate(lengths):
        k = min(2048, length)
        idx = out[row, :k].long()
        assert idx.unique().numel() == k
        got = logits[row, idx].float().sort().values
        want = logits[row, :length].float().topk(k).values.sort().values
        torch.testing.assert_close(got, want)


def test_takes_bf16_logits_alignment():
    from sglang.kernels.jit.csrc.dsa.kpool_topk_transform import takes_bf16_logits

    aligned = torch.empty(4, 65536, dtype=torch.bfloat16, device="cuda")
    assert takes_bf16_logits(aligned)
    unaligned = torch.empty(4, 65535, dtype=torch.bfloat16, device="cuda")
    assert not takes_bf16_logits(unaligned)
    fp32 = torch.empty(4, 65536, dtype=torch.float32, device="cuda")
    assert not takes_bf16_logits(fp32)
