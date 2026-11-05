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
"""Tests for the two-phase extensible KV cache sizing helpers."""

import pytest

from sglang.srt.mem_cache.extensible_kv_cache import (
    ExtensibleKVCacheConfig,
    ExtensibleKVCachePlan,
    all_ranks_support_extensible_kv_cache,
    commit_extensible_kv_cache,
    extensible_kv_cache_enabled,
    fallback_num_blocks,
    plan_extensible_kv_cache,
)
from sglang.srt.mem_cache.vmm_allocator import is_vmm_available


class _FakeBuffer:
    def __init__(self, reserved_size: int) -> None:
        self.reserved_size = reserved_size
        self.committed_bytes = 0
        self.freed = False

    def commit(self, num_bytes: int) -> int:
        if num_bytes > self.reserved_size:
            raise ValueError("exceeds reservation")
        self.committed_bytes = max(self.committed_bytes, num_bytes)
        return self.committed_bytes

    def free(self) -> None:
        self.freed = True


def _plan(bytes_per_block: int = 1024, max_num_blocks: int = 1024) -> ExtensibleKVCachePlan:
    return ExtensibleKVCachePlan(
        buffer=_FakeBuffer(bytes_per_block * max_num_blocks),
        bytes_per_block=bytes_per_block,
        max_num_blocks=max_num_blocks,
    )


def test_enabled_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True


def test_final_num_blocks_respects_cap_and_margin():
    plan = _plan()
    blocks = plan.final_num_blocks(free_bytes=plan.reserved_bytes)
    assert 0 <= blocks <= plan.max_num_blocks
    assert blocks < plan.max_num_blocks


def test_commit_is_idempotent_and_monotonic():
    plan = _plan()
    first = plan.commit(10)
    second = plan.commit(10)
    assert first == second
    third = plan.commit(5)
    assert third == first
    fourth = plan.commit(20)
    assert fourth >= first


def test_commit_never_exceeds_reservation():
    plan = _plan()
    committed = plan.commit(plan.max_num_blocks + 100)
    assert committed <= plan.max_num_blocks


def test_commit_extensible_uses_supplied_free_bytes():
    plan = _plan()
    committed = commit_extensible_kv_cache(plan, free_bytes=plan.reserved_bytes)
    assert committed == plan.committed_num_blocks()
    assert committed <= plan.max_num_blocks


def test_fallback_num_blocks_matches_estimate():
    assert fallback_num_blocks(1024, 4096, 10) == 4
    assert fallback_num_blocks(1024, 4096, 2) == 2
    with pytest.raises(ValueError):
        fallback_num_blocks(0, 4096, 2)


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1024, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_returns_none_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setenv("SGLANG_DISABLE_VMM", "1")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1024, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_all_ranks_support_without_group_returns_local():
    assert all_ranks_support_extensible_kv_cache(True) is True
    assert all_ranks_support_extensible_kv_cache(False) is False


def test_plan_free_releases_buffer():
    plan = _plan()
    plan.free()
    assert plan.buffer.freed is True


@pytest.mark.skipif(not is_vmm_available(), reason="VMM unavailable")
def test_plan_roundtrip_with_real_vmm(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    plan = plan_extensible_kv_cache(config)
    assert plan is not None
    try:
        committed = plan.commit(2)
        assert committed == 2
        assert plan.committed_num_blocks() == 2
    finally:
        plan.free()
