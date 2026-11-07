import pytest
import torch

pytest.importorskip("triton", reason="triton is required for warmup tests")

from sglang.kernels.ops.attention.fla.l2norm import (
    _l2norm_block_dim,
    l2norm_fwd,
    l2norm_fwd_block_warmup_inputs,
    triton_ptr_range,
)
from sglang.srt.layers.attention.dsa.kpool_plan import (
    _KPOOL_WARMUP_REGISTERED,
    build_kpool_compress_warmup_meta,
    register_kpool_compress_warmup,
    register_kpool_tail_seed_warmup,
)

# Importing the KDA Triton backend pulls in the full FLA kernel stack, which
# needs a usable Triton toolchain. Keep the import lazy so the pure-metadata
# tests below still run on CPU-only CI where the backend cannot be imported.
try:
    from sglang.srt.layers.attention.linear.kernels.kda_triton import (
        TritonKDAKernel,
    )

    _HAS_KDA_TRITON = True
except Exception:  # pragma: no cover - environment dependent
    TritonKDAKernel = None  # type: ignore[assignment]
    _HAS_KDA_TRITON = False


requires_kda_triton = pytest.mark.skipif(
    not _HAS_KDA_TRITON,
    reason="KDA Triton backend is unavailable in this environment",
)


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("ptr_range", [None, 1 << 32])
def test_l2norm_block_dim_matches_runtime(dim, dtype, ptr_range):
    block_dim = _l2norm_block_dim(dtype.itemsize, dim)
    assert block_dim >= dim
    assert block_dim == min(65536 // dtype.itemsize, 1 << (dim - 1).bit_length())


def test_l2norm_fwd_unchanged_by_block_dim_helper():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA device for l2norm_fwd numerical check")
    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    y = l2norm_fwd(x)
    ref = x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    torch.testing.assert_close(y.float(), ref, atol=1e-2, rtol=1e-2)


def test_triton_ptr_range_reads_ptr_range_attribute():
    t = torch.empty(64, dtype=torch.uint8)
    assert triton_ptr_range(t) == t.untyped_storage().size()
    t.ptr_range = lambda: 1 << 32
    assert triton_ptr_range(t) == 1 << 32


def test_triton_ptr_range_tolerates_missing_storage():
    class _NoStorage:
        pass

    assert triton_ptr_range(_NoStorage()) == 0

    class _BrokenStorage:
        def untyped_storage(self):
            raise RuntimeError("no storage")

    assert triton_ptr_range(_BrokenStorage()) == 0


@pytest.mark.parametrize("ptr_range_bytes", [0, 1 << 32])
def test_l2norm_warmup_inputs_carry_ptr_range(ptr_range_bytes):
    meta = l2norm_fwd_block_warmup_inputs(
        dtype=torch.bfloat16, dim=128, ptr_range_bytes=ptr_range_bytes
    )
    assert meta["x"].ptr_range_bytes == ptr_range_bytes
    assert meta["y"].ptr_range_bytes == ptr_range_bytes
    assert meta["x"].shape == (1, 128)
    assert meta["block_dim"] == _l2norm_block_dim(2, 128)


def test_kpool_warmup_meta_carries_real_ptr_range():
    class _FakeKCache:
        def __init__(self, kv_cache):
            self.kv_cache = kv_cache

    kv_cache = torch.empty((4, 64, 128), dtype=torch.uint8)
    kv_cache.ptr_range = lambda: 1 << 32
    meta = build_kpool_compress_warmup_meta(
        k_cache=_FakeKCache(kv_cache), pool_size=4, head_dim=128, round_scale=False
    )
    assert meta["buf_ptr_range_bytes"] == 1 << 32
    assert meta["page_size"] == 64
    assert meta["head_dim"] == 128


def test_kpool_warmup_registration_is_idempotent():
    class _FakeKCache:
        def __init__(self, kv_cache):
            self.kv_cache = kv_cache

    _KPOOL_WARMUP_REGISTERED.clear()
    kv_cache = torch.empty((4, 64, 128), dtype=torch.uint8)
    k_cache = _FakeKCache(kv_cache)
    assert register_kpool_compress_warmup(
        k_cache=k_cache, pool_size=4, head_dim=128, round_scale=False
    )
    assert not register_kpool_compress_warmup(
        k_cache=k_cache, pool_size=4, head_dim=128, round_scale=False
    )
    _KPOOL_WARMUP_REGISTERED.clear()


def test_kpool_tail_seed_warmup_registration_is_idempotent():
    _KPOOL_WARMUP_REGISTERED.clear()
    tail_cache = torch.empty((4, 64, 128), dtype=torch.uint8)
    assert register_kpool_tail_seed_warmup(
        tail_cache=tail_cache, kpool=4, head_dim=128
    )
    assert not register_kpool_tail_seed_warmup(
        tail_cache=tail_cache, kpool=4, head_dim=128
    )
    _KPOOL_WARMUP_REGISTERED.clear()


@requires_kda_triton
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA device for l2norm warmup launch",
)
def test_kda_l2norm_warmup_registration_is_idempotent():
    TritonKDAKernel._l2norm_warmup_registered.clear()
    TritonKDAKernel._l2norm_warmup_ptr_range.clear()
    try:
        assert TritonKDAKernel.register_l2norm_warmup(torch.bfloat16, 128)
        assert not TritonKDAKernel.register_l2norm_warmup(torch.bfloat16, 128)
        assert (torch.bfloat16, 128) in TritonKDAKernel._l2norm_warmup_ptr_range
    finally:
        TritonKDAKernel._l2norm_warmup_registered.clear()
        TritonKDAKernel._l2norm_warmup_ptr_range.clear()


@requires_kda_triton
def test_kda_l2norm_warmup_degrades_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    TritonKDAKernel._l2norm_warmup_registered.clear()
    TritonKDAKernel._l2norm_warmup_ptr_range.clear()
    try:
        assert not TritonKDAKernel.register_l2norm_warmup(torch.bfloat16, 128)
        assert not TritonKDAKernel._l2norm_warmup_registered
        assert not TritonKDAKernel._l2norm_warmup_ptr_range
    finally:
        TritonKDAKernel._l2norm_warmup_registered.clear()
        TritonKDAKernel._l2norm_warmup_ptr_range.clear()
