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
"""Tests for the two-phase extensible KV cache sizing helpers.

These tests exercise the pure sizing/fallback logic and do not require a GPU
or a driver VMM API. The VMM-backed path is covered by
``test_vmm_allocator.py`` and is skipped when VMM is unavailable.
"""

import os
from unittest import mock

import pytest

from sglang.srt.mem_cache.extensible_kv_cache import (
    KV_CACHE_MARGIN_FLOOR_BYTES,
    KV_CACHE_MARGIN_FRACTION,
    ExtensibleKVCacheConfig,
    ExtensibleKVCachePlan,
    all_ranks_support_extensible_kv_cache,
    commit_extensible_kv_cache,
    extensible_kv_cache_enabled,
    fallback_num_blocks,
    plan_extensible_kv_cache,
)


class _FakeBuffer:
    """Stand-in for VmmBuffer that records commits without touching a driver."""

    def __init__(self, reserved_size: int) -> None:
        self.reserved_size = reserved_size
        self.committed_bytes = 0
        self.freed = False

    def commit(self, num_bytes: int) -> int:
        if num_bytes > self.reserved_size:
            raise ValueError("commit exceeds reservation")
        self.committed_bytes = max(self.committed_bytes, num_bytes)
        return self.committed_bytes

    def free(self) -> None:
        self.freed = True


def _make_plan(
    bytes_per_block: int = 1024,
    max_num_blocks: int = 100,
    transient_peak_bytes: int = 0,
) -> ExtensibleKVCachePlan:
    return ExtensibleKVCachePlan(
        buffer=_FakeBuffer(bytes_per_block * max_num_blocks),
        bytes_per_block=bytes_per_block,
        max_num_blocks=max_num_blocks,
        transient_peak_bytes=transient_peak_bytes,
    )


def test_enabled_defaults_off():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", None)
        assert extensible_kv_cache_enabled() is False


def test_plan_returns_none_when_disabled():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", None)
        config = ExtensibleKVCacheConfig(
            bytes_per_block=1024, max_num_blocks=10, device_index=0
        )
        assert plan_extensible_kv_cache(config) is None


def test_final_num_blocks_respects_upper_bound():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    # Plenty of free memory: capped by the reservation.
    assert plan.final_num_blocks(1 << 40) == 100


def test_final_num_blocks_subtracts_margin():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=10_000)
    free = KV_CACHE_MARGIN_FLOOR_BYTES + 10 * 1024
    assert plan.final_num_blocks(free) == 10


def test_final_num_blocks_uses_transient_peak():
    transient = 4 * KV_CACHE_MARGIN_FLOOR_BYTES
    plan = _make_plan(
        bytes_per_block=1024, max_num_blocks=10_000, transient_peak_bytes=transient
    )
    free = int(transient * KV_CACHE_MARGIN_FRACTION) + 5 * 1024
    assert plan.final_num_blocks(free) == 5


def test_final_num_blocks_never_negative():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    assert plan.final_num_blocks(0) == 0


def test_commit_is_idempotent_and_monotonic():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    assert plan.commit(10) == 10
    assert plan.commit(10) == 10
    assert plan.commit(5) == 10
    assert plan.commit(20) == 20
    assert plan.committed_num_blocks() == 20


def test_commit_clamps_to_max():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=10)
    assert plan.commit(1000) == 10


def test_commit_rejects_negative():
    plan = _make_plan()
    with pytest.raises(ValueError):
        plan.commit(-1)


def test_commit_extensible_kv_cache_uses_measured_free():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=10_000)
    free = KV_CACHE_MARGIN_FLOOR_BYTES + 7 * 1024
    assert commit_extensible_kv_cache(plan, free_bytes=free) == 7


def test_fallback_num_blocks_matches_estimate():
    assert fallback_num_blocks(1024, 10 * 1024, 100) == 10
    assert fallback_num_blocks(1024, 10 * 1024, 3) == 3
    assert fallback_num_blocks(1024, 0, 100) == 0


def test_fallback_num_blocks_rejects_bad_input():
    with pytest.raises(ValueError):
        fallback_num_blocks(0, 1024, 10)
    with pytest.raises(ValueError):
        fallback_num_blocks(1024, -1, 10)
    with pytest.raises(ValueError):
        fallback_num_blocks(1024, 1024, -1)


def test_all_ranks_support_without_group_returns_local():
    assert all_ranks_support_extensible_kv_cache(True) is True
    assert all_ranks_support_extensible_kv_cache(False) is False


def test_all_ranks_support_falls_back_when_any_rank_lacks_vmm():
    class _Group:
        def __init__(self, value: int) -> None:
            self.value = value

        def all_reduce(self, tensor, op=None):
            tensor.fill_(self.value)

    assert all_ranks_support_extensible_kv_cache(True, _Group(1)) is True
    assert all_ranks_support_extensible_kv_cache(True, _Group(0)) is False
