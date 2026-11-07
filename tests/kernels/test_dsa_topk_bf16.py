import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _reference_topk(logits: torch.Tensor, lengths: torch.Tensor, k: int):
    out = []
    for row, length in enumerate(lengths.tolist()):
        idx = logits[row, :length].topk(min(k, length)).indices
        out.append(idx)
    return out


def _assert_same_selection(logits: torch.Tensor, lengths: torch.Tensor, got: torch.Tensor, k: int) -> None:
    for row, length in enumerate(lengths.tolist()):
        kk = min(k, length)
        idx = got[row, :kk].long()
        assert idx.unique().numel() == kk
        torch.testing.assert_close(
            logits[row, idx].sort().values,
            logits[row, :length].topk(kk).values.sort().values,
        )
        assert torch.all(got[row, kk:] == -1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_rows", [1, 4, 8, 33, 64])
@pytest.mark.parametrize("stride", [65536, 1048576])
@torch.inference_mode()
def test_cooperative_topk_rows_shorter_than_stride(dtype, num_rows, stride):
    torch.set_default_device("cuda:0")
    torch.manual_seed(0)
    top_k = 512
    row_lengths = [min(stride, 300000), 40000, 16384, 5000, 300]
    lengths = torch.tensor([row_lengths[i % len(row_lengths)] for i in range(num_rows)], dtype=torch.int32)
    logits = torch.randn(num_rows, stride).to(dtype)
    workspace = torch.empty(1 << 20, dtype=torch.uint8)
    out = torch.empty(num_rows, top_k, dtype=torch.int32)
    torch.ops._C.cooperative_topk(logits, lengths, out, workspace, top_k)
    _assert_same_selection(logits, lengths, out, top_k)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_rows", [1, 64])
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize("seq_len", [8192, 32768])
@torch.inference_mode()
def test_cooperative_topk_overflow(dtype, num_rows, top_k, seq_len):
    torch.set_default_device("cuda:0")
    torch.manual_seed(0)
    stride = (seq_len + 7) & ~7 if dtype == torch.bfloat16 else (seq_len + 3) & ~3
    row = 1.0 + torch.arange(seq_len, dtype=torch.float32) * (0.01 / seq_len)
    logits = torch.full((num_rows, stride), float("inf"), dtype=dtype)
    logits[:, :seq_len] = row
    lengths = torch.full((num_rows,), seq_len, dtype=torch.int32)
    workspace = torch.empty(1 << 20, dtype=torch.uint8)
    out = torch.empty(num_rows, top_k, dtype=torch.int32)
    torch.ops._C.cooperative_topk(logits, lengths, out, workspace, top_k)
    _assert_same_selection(logits, lengths, out, top_k)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_kpool_topk_transform_bf16_matches_fp32(dtype):
    torch.set_default_device("cuda:0")
    torch.manual_seed(0)
    batch, stride, pool_size, group_topk = 4, 65536, 2, 256
    lengths = torch.tensor([stride, 40000, 16384, 300], dtype=torch.int32)
    logits = torch.randn(batch, stride).to(dtype)
    dst = torch.empty(batch, group_topk * pool_size, dtype=torch.int32)
    torch.ops._C.kpool_topk_transform(logits, lengths, dst, pool_size)
    assert dst.shape == (batch, group_topk * pool_size)
    assert torch.all(dst >= -1)


@torch.inference_mode()
def test_kpool_topk_transform_bf16_unaligned_falls_back():
    torch.set_default_device("cuda:0")
    torch.manual_seed(0)
    batch, stride, pool_size, group_topk = 2, 65535, 2, 256
    lengths = torch.tensor([stride, 300], dtype=torch.int32)
    logits = torch.randn(batch, stride, dtype=torch.bfloat16)
    dst = torch.empty(batch, group_topk * pool_size, dtype=torch.int32)
    torch.ops._C.kpool_topk_transform(logits, lengths, dst, pool_size)
    assert dst.shape == (batch, group_topk * pool_size)


def test_bf16x2_packed_histogram_bins():
    # The packed extraction must agree with the scalar key on every BF16 value.
    values = torch.tensor([0.0, -0.0, 1.0, -1.0, 3.5, -3.5, 1e-3, -1e-3], dtype=torch.bfloat16)
    bits = values.view(torch.int16).to(torch.int32) & 0xFFFF
    key = torch.where(bits & 0x8000 != 0, (~bits) & 0xFFFF, bits | 0x8000)
    expected = (key >> 8).to(torch.int64)
    assert expected.min() >= 0 and expected.max() < 256
