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

These tests exercise the pure sizing logic and the fallback contract without
requiring a GPU: the VMM reservation itself is covered by
test_vmm_allocator.py, which skips when the driver API is unavailable.
"""

import pytest

from sglang.srt.mem_cache import extensible_kv_cache as ekc


class _FakeBuffer:
    """Stand-in for VmmBuffer that records commits without touching a driver."""

    def __init__(self, reserved_size: int, device_index: int = 0) -> None:
        self.reserved_size = reserved_size
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
        # Monotonic and idempotent, like the real buffer.
        self.committed_bytes = max(self.committed_bytes, num_bytes)
        return self.committed_bytes

    def free(self) -> None:
        self.freed = True
        self.committed_bytes = 0


def _make_plan(bytes_per_block=1024, max_num_blocks=100, transient_peak_bytes=0):
    plan = ekc.ExtensibleKVCachePlan(
        buffer=_FakeBuffer(bytes_per_block * max_num_blocks),
        bytes_per_block=bytes_per_block,
        max_num_blocks=max_num_blocks,
        transient_peak_bytes=transient_peak_bytes,
    )
    return plan


def test_enabled_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert ekc.extensible_kv_cache_enabled() is False


def test_enabled_flag_reads_env(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert ekc.extensible_kv_cache_enabled() is True


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    cfg = ekc.ExtensibleKVCacheConfig(
        bytes_per_block=1024, max_num_blocks=10, device_index=0
    )
    assert ekc.plan_extensible_kv_cache(cfg) is None


def test_final_num_blocks_respects_cap_and_margin():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    # Free memory far above the cap: never exceed max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 100
    # Free memory below the floor margin: no blocks.
    assert plan.final_num_blocks(0) == 0


def test_final_num_blocks_uses_measured_free_memory():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    free = ekc.KV_CACHE_MARGIN_FLOOR_BYTES + 10 * 1024
    assert plan.final_num_blocks(free) == 10


def test_final_num_blocks_rejects_negative_free():
    plan = _make_plan()
    with pytest.raises(ValueError):
        plan.final_num_blocks(-1)


def test_commit_is_idempotent_and_monotonic():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    assert plan.commit(10) == 10
    assert plan.commit(10) == 10
    assert plan.commit(5) == 10
    assert plan.committed_num_blocks() == 10


def test_commit_clamps_to_max_blocks():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    assert plan.commit(1000) == 100


def test_commit_rejects_negative_blocks():
    plan = _make_plan()
    with pytest.raises(ValueError):
        plan.commit(-1)


def test_commit_after_free_raises():
    plan = _make_plan()
    plan.commit(1)
    plan.free()
    with pytest.raises(RuntimeError):
        plan.commit(1)


def test_commit_extensible_kv_cache_uses_supplied_free_bytes():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    free = ekc.KV_CACHE_MARGIN_FLOOR_BYTES + 7 * 1024
    assert ekc.commit_extensible_kv_cache(plan, free_bytes=free) == 7


def test_all_ranks_support_returns_local_without_group():
    assert ekc.all_ranks_support_extensible_kv_cache(True) is True
    assert ekc.all_ranks_support_extensible_kv_cache(False) is False


def test_all_ranks_support_uses_min_reduce():
    class _Group:
        def __init__(self, value: int) -> None:
            self.value = value

        def all_reduce(self, tensor, op=None):
            # Simulate a peer that does not support VMM.
            tensor.fill_(min(int(tensor.item()), self.value))

    assert ekc.all_ranks_support_extensible_kv_cache(True, _Group(0)) is False
    assert ekc.all_ranks_support_extensible_kv_cache(True, _Group(1)) is True
