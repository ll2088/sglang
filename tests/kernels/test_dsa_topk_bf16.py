import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _has_op(name: str) -> bool:
    try:
        return hasattr(torch.ops._C, name)
    except Exception:
        return False


requires_cooperative_topk = pytest.mark.skipif(
    not _has_op("cooperative_topk"), reason="torch.ops._C.cooperative_topk is not built"
)
requires_kpool_topk = pytest.mark.skipif(
    not _has_op("kpool_topk_transform"), reason="torch.ops._C.kpool_topk_transform is not built"
)


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


@requires_cooperative_topk
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


@requires_cooperative_topk
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


@requires_kpool_topk
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


@requires_kpool_topk
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


def _scalar_bf16_key(bits: int) -> int:
    # Mirrors convert_bf16_to_uint8: ordered 16-bit key of a BF16 bit pattern.
    return (~bits) & 0xFFFF if bits & 0x8000 else (bits | 0x8000)


def _packed_bf16x2_bins(w: int) -> tuple:
    # Mirrors convert_bf16x2_to_uint8x2: two BF16 scores packed low half first.
    lo = w & 0xFFFF
    hi = (w >> 16) & 0xFFFF
    return (_scalar_bf16_key(lo) >> 8, _scalar_bf16_key(hi) >> 8)


def test_bf16x2_packed_histogram_bins():
    # The packed extraction must agree with the scalar key on every BF16 value.
    values = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 3.5, -3.5, 1e-3, -1e-3, 65504.0, -65504.0],
        dtype=torch.bfloat16,
    )
    bits = (values.view(torch.int16).to(torch.int32) & 0xFFFF).tolist()
    for i in range(0, len(bits) - 1, 2):
        w = bits[i] | (bits[i + 1] << 16)
        lo, hi = _packed_bf16x2_bins(w)
        assert lo == _scalar_bf16_key(bits[i]) >> 8
        assert hi == _scalar_bf16_key(bits[i + 1]) >> 8
        assert 0 <= lo < 256 and 0 <= hi < 256


def test_bf16x2_packed_histogram_bins_ordering():
    # Monotonicity: a larger BF16 score must never map to a smaller bin.
    values = torch.tensor(
        [-65504.0, -3.5, -1.0, -0.0, 0.0, 1e-3, 1.0, 3.5, 65504.0],
        dtype=torch.bfloat16,
    )
    bits = (values.view(torch.int16).to(torch.int32) & 0xFFFF).tolist()
    bins = [_scalar_bf16_key(b) >> 8 for b in bits]
    assert bins == sorted(bins)
