from typing import Any, Optional

import torch

from sglang.kernels.ops.attention.fla.l2norm import triton_ptr_range

# Idempotent registration bookkeeping for kpool warmup. Keyed by the identity
# of the cache object plus the shape parameters so repeated construction of the
# same indexer does not re-issue warmup launches.
_KPOOL_WARMUP_REGISTERED: set[tuple[int, int, int, bool]] = set()


def build_kpool_compress_warmup_meta(
    *, k_cache: Any, pool_size: int, head_dim: int, round_scale: bool
) -> dict[str, Any]:
    """Describe the kpool compress warmup inputs from the live KV cache.

    The compressed outputs alias views of the KV cache, so their pointer range
    must be derived from the real cache storage rather than assumed small.
    """
    kv_cache = k_cache.kv_cache
    page_size = kv_cache.shape[1]
    buf_numel_per_page = kv_cache.stride(0)
    # Derive the pointer range from the real cache storage. ``triton_ptr_range``
    # already tolerates fake tensors and missing storage by returning 0, which
    # keeps the optimistic <2GiB default instead of emitting a bogus key.
    buf_ptr_range_bytes = triton_ptr_range(kv_cache)
    return dict(
        page_size=page_size,
        buf_numel_per_page=buf_numel_per_page,
        head_dim=head_dim,
        pool_size=pool_size,
        round_scale=round_scale,
        buf_ptr_range_bytes=buf_ptr_range_bytes,
    )


def register_kpool_compress_warmup(
    *, k_cache: Any, pool_size: int, head_dim: int, round_scale: bool
) -> bool:
    """Idempotently register kpool compress warmup.

    Returns True when a new registration was recorded, False when the same
    cache/shape combination was already registered. The actual warmup launch is
    owned by the dispatcher; this only records the derived metadata so the
    pointer range is not lost.
    """
    key = (id(k_cache), pool_size, head_dim, round_scale)
    if key in _KPOOL_WARMUP_REGISTERED:
        return False
    build_kpool_compress_warmup_meta(
        k_cache=k_cache, pool_size=pool_size, head_dim=head_dim, round_scale=round_scale
    )
    _KPOOL_WARMUP_REGISTERED.add(key)
    return True


def register_kpool_tail_seed_warmup(
    *, tail_cache: Any, kpool: int, head_dim: int
) -> bool:
    """Idempotently register kpool tail-seed warmup."""
    key = (id(tail_cache), kpool, head_dim, False)
    if key in _KPOOL_WARMUP_REGISTERED:
        return False
    _KPOOL_WARMUP_REGISTERED.add(key)
    return True
