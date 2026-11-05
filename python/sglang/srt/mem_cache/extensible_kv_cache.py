# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Two-phase KV cache sizing helpers.

This module implements the measurement-based sizing flow from vLLM's
extensible KV cache in a minimal, opt-in form:

1. Reserve address space for an upper bound of KV cache blocks.
2. After warmup, measure the free device memory and compute the final block
   count from it.
3. Commit physical pages for the final block count into the reservation.

If VMM is unavailable or any step fails, callers fall back to the existing
estimate-based allocation path. The helpers here never mutate global state and
are safe to call repeatedly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.mem_cache.vmm_allocator import (
    VmmBuffer,
    VmmUnavailableError,
    vmm_unavailable_reason,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ExtensibleKVCacheConfig",
    "ExtensibleKVCachePlan",
    "extensible_kv_cache_enabled",
    "plan_extensible_kv_cache",
    "commit_extensible_kv_cache",
]

# Headroom kept free after sizing from measured memory. The fraction scales
# with the measured transient peak; the floor covers small models.
KV_CACHE_MARGIN_FLOOR_BYTES = 1 << 30
KV_CACHE_MARGIN_FRACTION = 0.75


def extensible_kv_cache_enabled() -> bool:
    """Whether the opt-in extensible KV cache path is enabled."""
    return os.environ.get("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "0") == "1"


@dataclass(frozen=True)
class ExtensibleKVCacheConfig:
    """Inputs for the two-phase sizing flow."""

    bytes_per_block: int
    max_num_blocks: int
    device_index: int
    # Measured transient peak (bytes) observed during warmup. Used to size the
    # headroom that is left uncommitted.
    transient_peak_bytes: int = 0


@dataclass
class ExtensibleKVCachePlan:
    """Result of the reservation phase."""

    buffer: VmmBuffer
    bytes_per_block: int
    max_num_blocks: int
    transient_peak_bytes: int = 0

    @property
    def reserved_bytes(self) -> int:
        return self.buffer.reserved_size

    def final_num_blocks(self, free_bytes: int) -> int:
        """Compute the final block count from measured free memory.

        The result never exceeds ``max_num_blocks`` and leaves a measured
        headroom uncommitted. The headroom is derived from the measured
        transient peak when available, falling back to a fraction of the
        free memory so small models still keep a floor.
        """
        if free_bytes < 0:
            raise ValueError("free_bytes must be non-negative")
        if self.bytes_per_block <= 0:
            raise ValueError("bytes_per_block must be positive")
        margin = max(
            KV_CACHE_MARGIN_FLOOR_BYTES,
            int(self.transient_peak_bytes * KV_CACHE_MARGIN_FRACTION),
        )
        usable = max(free_bytes - margin, 0)
        blocks = usable // self.bytes_per_block
        return min(blocks, self.max_num_blocks)

    def commit(self, num_blocks: int) -> int:
        """Commit physical pages for ``num_blocks`` blocks.

        Returns the number of blocks actually backed. Idempotent and monotonic:
        committing a smaller count after a larger one is a no-op, and a failed
        commit leaves the plan at its previous committed count.
        """
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        num_blocks = min(num_blocks, self.max_num_blocks)
        self.buffer.commit(num_blocks * self.bytes_per_block)
        return self.buffer.committed_bytes // self.bytes_per_block

    def committed_num_blocks(self) -> int:
        """Number of blocks whose physical pages are currently mapped."""
        return self.buffer.committed_bytes // self.bytes_per_block

    def free(self) -> None:
        self.buffer.free()


def plan_extensible_kv_cache(
    config: ExtensibleKVCacheConfig,
) -> Optional[ExtensibleKVCachePlan]:
    """Reserve address space for the upper bound of KV cache blocks.

    Returns ``None`` when VMM is unavailable or the reservation fails, so the
    caller can fall back to the existing allocation path.
    """
    if not extensible_kv_cache_enabled():
        return None
    reason = vmm_unavailable_reason()
    if reason is not None:
        logger.info("Extensible KV cache disabled: %s", reason)
        return None
    if config.bytes_per_block <= 0 or config.max_num_blocks <= 0:
        return None
    try:
        buffer = VmmBuffer(
            config.bytes_per_block * config.max_num_blocks,
            config.device_index,
        )
    except VmmUnavailableError as exc:
        logger.info("Extensible KV cache reservation failed: %s", exc)
        return None
    return ExtensibleKVCachePlan(
        buffer=buffer,
        bytes_per_block=config.bytes_per_block,
        max_num_blocks=config.max_num_blocks,
        transient_peak_bytes=config.transient_peak_bytes,
    )


def commit_extensible_kv_cache(
    plan: ExtensibleKVCachePlan,
    free_bytes: Optional[int] = None,
) -> int:
    """Measure free memory (if not supplied) and commit the final block count.

    Returns the number of committed blocks. Raises ``VmmUnavailableError`` on
    mapping failure so the caller can fall back. Repeated calls are idempotent
    and monotonic: a later call never commits fewer blocks than an earlier one.
    """
    if free_bytes is None:
        free_bytes, _ = torch.cuda.mem_get_info(plan.buffer.device_index)
    num_blocks = plan.final_num_blocks(free_bytes)
    return plan.commit(num_blocks)


def fallback_num_blocks(
    bytes_per_block: int,
    free_bytes: int,
    max_num_blocks: int,
) -> int:
    """Estimate-based block count used when the VMM path is unavailable.

    This mirrors the pre-change sizing behaviour: the whole measured free
    memory is divided by the per-block size, capped by the upper bound. It is
    kept separate from :class:`ExtensibleKVCachePlan` so the fallback path can
    be exercised without a driver or a reservation.
    """
    if bytes_per_block <= 0:
        raise ValueError("bytes_per_block must be positive")
    if free_bytes < 0:
        raise ValueError("free_bytes must be non-negative")
    if max_num_blocks < 0:
        raise ValueError("max_num_blocks must be non-negative")
    return min(free_bytes // bytes_per_block, max_num_blocks)


def all_ranks_support_extensible_kv_cache(
    local_supported: bool,
    group=None,
) -> bool:
    """Return True only when every rank in ``group`` supports the VMM path.

    The feature switch must be globally consistent: if one rank falls back to
    the estimate-based path while another reserves VMM memory, the block count
    all-reduce would mismatch and deadlock. ``group`` is any object exposing
    ``all_reduce`` on a tensor; when it is None the local answer is returned.
    """
    if group is None:
        return local_supported
    flag = torch.tensor(
        [1 if local_supported else 0],
        dtype=torch.int32,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    group.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item())
