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

import pytest

from sglang.srt.mem_cache import extensible_kv_cache as ekc
from sglang.srt.mem_cache.extensible_kv_cache import (
    ExtensibleKVCacheConfig,
    ExtensibleKVCachePlan,
    commit_extensible_kv_cache,
    extensible_kv_cache_enabled,
    plan_extensible_kv_cache,
)


class _FakeBuffer:
    """In-memory stand-in for VmmBuffer so sizing logic is testable on CPU."""

    def __init__(self, max_bytes: int, device_index: int = 0) -> None:
        self.reserved_size = max_bytes
        self.device_index = device_index
        self.committed_bytes = 0
        self.freed = False

    def commit(self, num_bytes: int) -> int:
        if self.freed:
            raise RuntimeError("freed")
        if num_bytes < 0:
            raise ValueError("negative")
        if num_bytes > self.reserved_size:
            raise ValueError("exceeds reservation")
        if num_bytes > self.committed_bytes:
            self.committed_bytes = num_bytes
        return self.committed_bytes

    def free(self) -> None:
        self.freed = True
        self.committed_bytes = 0


def _make_plan(bytes_per_block: int = 1024, max_num_blocks: int = 100):
    buffer = _FakeBuffer(bytes_per_block * max_num_blocks)
    return ExtensibleKVCachePlan(
        buffer=buffer,
        bytes_per_block=bytes_per_block,
        max_num_blocks=max_num_blocks,
    )


def test_enabled_defaults_to_off(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=1024, max_num_blocks=10, device_index=0
    )
    assert plan_extensible_kv_cache(cfg) is None


def test_plan_returns_none_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: "no driver")
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=1024, max_num_blocks=10, device_index=0
    )
    assert plan_extensible_kv_cache(cfg) is None


def test_final_num_blocks_respects_margin_and_cap():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    # Free memory below the floor margin yields zero blocks.
    assert plan.final_num_blocks(0) == 0
    # Plenty of memory is capped at the reservation.
    assert plan.final_num_blocks(1 << 40) == 100
    # A mid-range value leaves the floor margin uncommitted.
    free = ekc.KV_CACHE_MARGIN_FLOOR_BYTES + 10 * 1024
    assert plan.final_num_blocks(free) == 10


def test_final_num_blocks_uses_transient_peak():
    buffer = _FakeBuffer(1024 * 100)
    plan = ExtensibleKVCachePlan(
        buffer=buffer,
        bytes_per_block=1024,
        max_num_blocks=100,
        transient_peak_bytes=4 * ekc.KV_CACHE_MARGIN_FLOOR_BYTES,
    )
    free = ekc.KV_CACHE_MARGIN_FLOOR_BYTES * 10
    expected_margin = int(4 * ekc.KV_CACHE_MARGIN_FLOOR_BYTES * ekc.KV_CACHE_MARGIN_FRACTION)
    expected = (free - expected_margin) // 1024
    assert plan.final_num_blocks(free) == expected


def test_commit_is_idempotent_and_monotonic():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    assert plan.commit(10) == 10
    assert plan.commit(10) == 10
    assert plan.commit(5) == 10
    assert plan.commit(20) == 20


def test_commit_clamps_to_reservation():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=10)
    assert plan.commit(1000) == 10


def test_commit_rejects_negative():
    plan = _make_plan()
    with pytest.raises(ValueError):
        plan.commit(-1)


def test_commit_extensible_uses_measured_free_bytes():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    free = ekc.KV_CACHE_MARGIN_FLOOR_BYTES + 7 * 1024
    assert commit_extensible_kv_cache(plan, free_bytes=free) == 7


def test_free_releases_buffer():
    plan = _make_plan()
    plan.commit(5)
    plan.free()
    assert plan.buffer.freed is True


def test_fallback_plan_is_none_and_caller_keeps_estimate():
    """When planning fails the caller must keep the estimate-based path."""
    plan = plan_extensible_kv_cache(
        ExtensibleKVCacheConfig(bytes_per_block=0, max_num_blocks=0, device_index=0)
    )
    assert plan is None
