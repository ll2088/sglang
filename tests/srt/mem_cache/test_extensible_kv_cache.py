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
"""Tests for the two-phase KV cache sizing helpers.

Covers the measurement-based block count, commit idempotency, the fallback
estimate path and the all-rank consistency check. No GPU is required: the
plan is built over a fake buffer so the sizing arithmetic is tested directly.
"""

import pytest
import torch

from sglang.srt.mem_cache import extensible_kv_cache as ekc
from sglang.srt.mem_cache.extensible_kv_cache import (
    ExtensibleKVCacheConfig,
    ExtensibleKVCachePlan,
    all_ranks_support_extensible_kv_cache,
    commit_extensible_kv_cache,
    extensible_kv_cache_enabled,
    fallback_num_blocks,
    plan_extensible_kv_cache,
)


class _FakeBuffer:
    def __init__(self, max_bytes: int, granularity: int = 2 << 20) -> None:
        self.granularity = granularity
        self.reserved_size = ((max_bytes + granularity - 1) // granularity) * granularity
        self.device_index = 0
        self._committed = 0
        self.freed = False

    @property
    def committed_bytes(self) -> int:
        return self._committed

    def commit(self, num_bytes: int) -> int:
        if self.freed:
            raise RuntimeError("freed")
        if num_bytes > self.reserved_size:
            raise ValueError("exceeds reservation")
        target = ((num_bytes + self.granularity - 1) // self.granularity) * self.granularity
        self._committed = max(self._committed, target)
        return self._committed

    def free(self) -> None:
        self.freed = True
        self._committed = 0


def _plan(bytes_per_block: int, max_blocks: int, transient: int = 0):
    return ExtensibleKVCachePlan(
        buffer=_FakeBuffer(bytes_per_block * max_blocks),
        bytes_per_block=bytes_per_block,
        max_num_blocks=max_blocks,
        transient_peak_bytes=transient,
    )


def test_enabled_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert not extensible_kv_cache_enabled()
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled()


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(cfg) is None


def test_final_num_blocks_respects_margin_and_cap():
    plan = _plan(bytes_per_block=1 << 20, max_blocks=100)
    # Free memory below the floor margin yields zero blocks.
    assert plan.final_num_blocks(0) == 0
    # Plenty of memory is capped by max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 100
    # A measured transient peak raises the margin proportionally.
    plan2 = _plan(bytes_per_block=1 << 20, max_blocks=100, transient=4 << 30)
    assert plan2.final_num_blocks(8 << 30) < plan.final_num_blocks(8 << 30)


def test_commit_is_idempotent_and_monotonic():
    plan = _plan(bytes_per_block=1 << 20, max_blocks=16)
    assert plan.commit(4) == 4
    assert plan.commit(4) == 4
    assert plan.commit(2) == 4  # never shrinks
    assert plan.commit(100) == 16  # capped by max_num_blocks
    assert plan.committed_num_blocks() == 16


def test_commit_rejects_negative():
    plan = _plan(bytes_per_block=1 << 20, max_blocks=4)
    with pytest.raises(ValueError):
        plan.commit(-1)


def test_commit_extensible_uses_supplied_free_bytes():
    plan = _plan(bytes_per_block=1 << 20, max_blocks=64)
    blocks = commit_extensible_kv_cache(plan, free_bytes=8 << 30)
    assert blocks == plan.committed_num_blocks()
    assert blocks <= plan.max_num_blocks


def test_fallback_matches_estimate_path():
    assert fallback_num_blocks(1 << 20, 8 << 30, 64) == 64
    assert fallback_num_blocks(1 << 20, 0, 64) == 0
    with pytest.raises(ValueError):
        fallback_num_blocks(0, 1, 1)
    with pytest.raises(ValueError):
        fallback_num_blocks(1, -1, 1)


def test_all_ranks_consistency_without_group():
    assert all_ranks_support_extensible_kv_cache(True) is True
    assert all_ranks_support_extensible_kv_cache(False) is False


class _MinGroup:
    """Stand-in for a process group that reduces with MIN."""

    def __init__(self, values):
        self.values = list(values)

    def all_reduce(self, tensor, op=None):
        assert op is torch.distributed.ReduceOp.MIN
        tensor.fill_(min(self.values))


def test_all_ranks_consistency_falls_back_when_any_rank_unsupported():
    group = _MinGroup([1, 0, 1])
    assert all_ranks_support_extensible_kv_cache(True, group) is False
    group = _MinGroup([1, 1, 1])
    assert all_ranks_support_extensible_kv_cache(True, group) is True
