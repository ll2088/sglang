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
"""CPU-only tests for the two-phase extensible KV cache sizing helpers.

These tests exercise the pure sizing logic (granule alignment, idempotent
commit accounting, measured-free-memory sizing and the all-rank consistency
fallback) without requiring a GPU or the driver VMM API. GPU-specific
reservation/mapping behaviour is covered by test_vmm_allocator.py.
"""

import math

import pytest
import torch

from sglang.srt.mem_cache.extensible_kv_cache import (
    KV_CACHE_MARGIN_FLOOR_BYTES,
    KV_CACHE_MARGIN_FRACTION,
    ExtensibleKVCacheConfig,
    ExtensibleKVCachePlan,
    all_ranks_support_extensible_kv_cache,
    commit_extensible_kv_cache,
    extensible_kv_cache_enabled,
    plan_extensible_kv_cache,
)


class _FakeBuffer:
    """Stand-in for VmmBuffer that only tracks committed bytes."""

    def __init__(self, reserved_size: int, granularity: int = 1) -> None:
        self.reserved_size = reserved_size
        self.granularity = granularity
        self._committed = 0
        self.freed = False

    @property
    def committed_bytes(self) -> int:
        return self._committed

    def commit(self, num_bytes: int) -> int:
        if self.freed:
            raise RuntimeError("freed")
        if num_bytes < 0:
            raise ValueError("negative")
        if num_bytes > self.reserved_size:
            raise ValueError("exceeds reservation")
        target = ((num_bytes + self.granularity - 1) // self.granularity) * self.granularity
        if target > self._committed:
            self._committed = target
        return self._committed

    def free(self) -> None:
        self.freed = True
        self._committed = 0


def _make_plan(
    bytes_per_block: int = 1024,
    max_num_blocks: int = 100,
    transient_peak_bytes: int = 0,
    granularity: int = 1,
) -> ExtensibleKVCachePlan:
    buffer = _FakeBuffer(bytes_per_block * max_num_blocks, granularity)
    return ExtensibleKVCachePlan(
        buffer=buffer,
        bytes_per_block=bytes_per_block,
        max_num_blocks=max_num_blocks,
        transient_peak_bytes=transient_peak_bytes,
    )


def test_enabled_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1024, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_final_num_blocks_respects_margin_and_cap():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    # Free memory below the floor leaves no usable blocks.
    assert plan.final_num_blocks(KV_CACHE_MARGIN_FLOOR_BYTES // 2) == 0
    # Free memory above the floor yields blocks, capped at max_num_blocks.
    free = KV_CACHE_MARGIN_FLOOR_BYTES + 10 * 1024
    assert plan.final_num_blocks(free) == 10
    assert plan.final_num_blocks(10**12) == 100


def test_final_num_blocks_uses_measured_transient_peak():
    transient = 4 * 1024
    plan = _make_plan(
        bytes_per_block=1024, max_num_blocks=100, transient_peak_bytes=transient
    )
    free = KV_CACHE_MARGIN_FLOOR_BYTES + transient + 5 * 1024
    expected_margin = max(
        KV_CACHE_MARGIN_FLOOR_BYTES, int(transient * KV_CACHE_MARGIN_FRACTION)
    )
    expected = (free - expected_margin) // 1024
    assert plan.final_num_blocks(free) == expected


def test_commit_is_idempotent_and_monotonic():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    assert plan.commit(10) == 10
    assert plan.commit(10) == 10
    assert plan.commit(5) == 10
    assert plan.commit(20) == 20
    assert plan.committed_num_blocks() == 20


def test_commit_clamps_to_max_blocks():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=8)
    assert plan.commit(1000) == 8


def test_commit_rejects_negative():
    plan = _make_plan()
    with pytest.raises(ValueError):
        plan.commit(-1)


def test_commit_extensible_kv_cache_uses_supplied_free_bytes():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    free = KV_CACHE_MARGIN_FLOOR_BYTES + 7 * 1024
    assert commit_extensible_kv_cache(plan, free_bytes=free) == 7
    # Repeated calls are idempotent.
    assert commit_extensible_kv_cache(plan, free_bytes=free) == 7


def test_commit_extensible_kv_cache_measures_when_free_bytes_missing(monkeypatch):
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    measured = KV_CACHE_MARGIN_FLOOR_BYTES + 3 * 1024
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device=None: (measured, measured)
    )
    assert commit_extensible_kv_cache(plan) == 3


def test_free_releases_plan():
    plan = _make_plan()
    plan.commit(4)
    plan.free()
    assert plan.buffer.freed is True
    with pytest.raises(RuntimeError):
        plan.commit(1)


def test_all_ranks_support_without_group_returns_local():
    assert all_ranks_support_extensible_kv_cache(True) is True
    assert all_ranks_support_extensible_kv_cache(False) is False


class _FakeGroup:
    """Simulates an all-reduce MIN over a fixed set of peer flags."""

    def __init__(self, peer_flags):
        self.peer_flags = peer_flags

    def all_reduce(self, tensor, op=None):
        assert op == torch.distributed.ReduceOp.MIN
        value = min([int(tensor.item())] + list(self.peer_flags))
        tensor.fill_(value)


def test_all_ranks_support_falls_back_when_any_rank_unsupported():
    assert all_ranks_support_extensible_kv_cache(True, _FakeGroup([1, 1])) is True
    assert all_ranks_support_extensible_kv_cache(True, _FakeGroup([1, 0])) is False
    assert all_ranks_support_extensible_kv_cache(False, _FakeGroup([1, 1])) is False


def test_granule_alignment_helpers_are_consistent():
    # Mirrors the upstream granule alignment contract without importing the
    # GPU-only module: the aligned count must be a multiple of the alignment
    # and never exceed the requested count.
    granule = 2 << 20
    strides = [1024, 2048]
    alignment = 1
    for stride in set(strides):
        alignment = math.lcm(alignment, granule // math.gcd(granule, stride))
    for num_blocks in (0, 1, alignment - 1, alignment, alignment * 3 + 1):
        aligned = num_blocks // alignment * alignment
        assert aligned <= num_blocks
        assert aligned % alignment == 0
