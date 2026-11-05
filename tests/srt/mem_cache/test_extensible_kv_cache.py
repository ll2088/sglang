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
"""Tests for the two-phase KV cache sizing helpers and their fallback."""

from __future__ import annotations

import pytest

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
    def __init__(self, reserved_size: int):
        self.reserved_size = reserved_size
        self.committed_bytes = 0
        self.freed = False
        self.device_index = 0

    def commit(self, num_bytes: int) -> int:
        if num_bytes > self.reserved_size:
            raise ValueError("exceeds reservation")
        self.committed_bytes = max(self.committed_bytes, num_bytes)
        return self.committed_bytes

    def free(self) -> None:
        self.freed = True


def _plan(bytes_per_block: int = 1024, max_num_blocks: int = 100, peak: int = 0):
    return ExtensibleKVCachePlan(
        buffer=_FakeBuffer(bytes_per_block * max_num_blocks),
        bytes_per_block=bytes_per_block,
        max_num_blocks=max_num_blocks,
        transient_peak_bytes=peak,
    )


def test_enabled_defaults_off(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert not extensible_kv_cache_enabled()


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
def test_enabled_accepts_truthy(monkeypatch, value):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", value)
    assert extensible_kv_cache_enabled()


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_enabled_rejects_falsy(monkeypatch, value):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", value)
    assert not extensible_kv_cache_enabled()


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1024, max_num_blocks=10, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_returns_none_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: "no driver")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1024, max_num_blocks=10, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_returns_none_on_reservation_failure(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: None)

    def _boom(*args, **kwargs):
        raise ekc.VmmUnavailableError("boom")

    monkeypatch.setattr(ekc, "VmmBuffer", _boom)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1024, max_num_blocks=10, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_final_num_blocks_respects_margin_and_cap():
    plan = _plan(bytes_per_block=1024, max_num_blocks=100)
    # Free memory below the floor leaves nothing usable.
    assert plan.final_num_blocks(0) == 0
    # Plenty of free memory is capped by max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 100


def test_final_num_blocks_uses_measured_peak():
    plan = _plan(bytes_per_block=1024, max_num_blocks=1000, peak=4 << 20)
    free = (4 << 20) + (4 << 20) * 3 // 4 + 1024 * 10
    assert plan.final_num_blocks(free) == 10


def test_final_num_blocks_rejects_negative_free():
    plan = _plan()
    with pytest.raises(ValueError):
        plan.final_num_blocks(-1)


def test_commit_is_idempotent_and_monotonic():
    plan = _plan(bytes_per_block=1024, max_num_blocks=100)
    assert plan.commit(10) == 10
    assert plan.commit(10) == 10
    assert plan.commit(5) == 10
    assert plan.commit(20) == 20
    assert plan.committed_num_blocks() == 20


def test_commit_clamps_to_max_blocks():
    plan = _plan(bytes_per_block=1024, max_num_blocks=10)
    assert plan.commit(1000) == 10


def test_commit_rejects_negative():
    plan = _plan()
    with pytest.raises(ValueError):
        plan.commit(-1)


def test_commit_extensible_uses_supplied_free_bytes():
    plan = _plan(bytes_per_block=1024, max_num_blocks=100)
    committed = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
    assert committed == 100


def test_commit_extensible_measures_free_memory(monkeypatch):
    plan = _plan(bytes_per_block=1024, max_num_blocks=100)
    monkeypatch.setattr(
        ekc.torch.cuda, "mem_get_info", lambda index: (1 << 40, 1 << 41)
    )
    assert commit_extensible_kv_cache(plan) == 100


def test_fallback_matches_estimate_path():
    assert fallback_num_blocks(1024, 10 * 1024, 100) == 10
    assert fallback_num_blocks(1024, 10 * 1024, 5) == 5
    assert fallback_num_blocks(1024, 0, 5) == 0


def test_fallback_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        fallback_num_blocks(0, 1024, 10)
    with pytest.raises(ValueError):
        fallback_num_blocks(1024, -1, 10)
    with pytest.raises(ValueError):
        fallback_num_blocks(1024, 1024, -1)


def test_all_ranks_support_without_group_returns_local():
    assert all_ranks_support_extensible_kv_cache(True)
    assert not all_ranks_support_extensible_kv_cache(False)


def test_all_ranks_support_uses_min_reduce(monkeypatch):
    class _Group:
        def __init__(self, value: int):
            self.value = value
            self.op = None

        def all_reduce(self, tensor, op=None):
            self.op = op
            tensor.fill_(self.value)

    group = _Group(0)
    assert not all_ranks_support_extensible_kv_cache(True, group)
    assert group.op is not None


def test_plan_free_releases_buffer():
    plan = _plan()
    plan.free()
    assert plan.buffer.freed
