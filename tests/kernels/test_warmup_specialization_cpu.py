import pytest
import torch

# CPU-only companion to test_warmup_specialization.py. It exercises the
# pointer-range metadata, warmup-input construction, and idempotent
# registration logic without requiring Triton, CUDA, or ROCm. This keeps the
# warmup specialization contract verifiable in the default CI environment.

from sglang.kernels.ops.attention.fla.l2norm import (
    _l2norm_block_dim,
    l2norm_fwd_block_warmup_inputs,
    l2norm_fwd_block_warmup_inputs_from_live,
    triton_ptr_range,
)
from sglang.srt.layers.attention.dsa.kpool_plan import (
    _KPOOL_WARMUP_REGISTERED,
    build_kpool_compress_warmup_meta,
    register_kpool_compress_warmup,
    register_kpool_tail_seed_warmup,
)


def test_l2norm_block_dim_matches_runtime():
    for dim in (64, 128, 256):
        for dtype in (torch.bfloat16, torch.float16):
            block_dim = _l2norm_block_dim(dtype.itemsize, dim)
            assert block_dim >= dim
            assert block_dim == min(65536 // dtype.itemsize, 1 << (dim - 1).bit_length())


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


def test_l2norm_warmup_inputs_from_live_uses_real_ptr_range():
    live = torch.empty(64, dtype=torch.uint8)
    live.ptr_range = lambda: 1 << 32
    meta = l2norm_fwd_block_warmup_inputs_from_live(
        dtype=torch.bfloat16, dim=128, live=live
    )
    assert meta["x"].ptr_range_bytes == 1 << 32
    assert meta["y"].ptr_range_bytes == 1 << 32


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


def test_kpool_warmup_meta_degrades_without_storage():
    class _FakeKCache:
        def __init__(self, kv_cache):
            self.kv_cache = kv_cache

    class _NoStorage:
        shape = (4, 64, 128)

        def stride(self, _dim):
            return 64 * 128

    meta = build_kpool_compress_warmup_meta(
        k_cache=_FakeKCache(_NoStorage()),
        pool_size=4,
        head_dim=128,
        round_scale=False,
    )
    assert meta["buf_ptr_range_bytes"] == 0
    assert meta["page_size"] == 64


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
