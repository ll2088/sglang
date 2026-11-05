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
"""Tests for the two-phase KV cache sizing helpers."""

from __future__ import annotations

import pytest

from sglang.srt.mem_cache import extensible_kv_cache as ekc


class _FakeBuffer:
    def __init__(self, reserved_size: int, granularity: int = 2 << 20) -> None:
        self.reserved_size = reserved_size
        self.granularity = granularity
        self.committed_bytes = 0
        self.freed = False

    def commit(self, num_bytes: int) -> int:
        if num_bytes > self.reserved_size:
            raise ValueError("exceeds reservation")
        target = ((num_bytes + self.granularity - 1) // self.granularity) * self.granularity
        if target > self.committed_bytes:
            self.committed_bytes = target
        return self.committed_bytes

    def free(self) -> None:
        self.freed = True
        self.committed_bytes = 0


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: None)
    return monkeypatch


def _plan(monkeypatch, max_blocks: int = 100, bytes_per_block: int = 1 << 20):
    created = {}

    def _make_buffer(size, device_index):
        buf = _FakeBuffer(size)
        created["buffer"] = buf
        return buf

    monkeypatch.setattr(ekc, "VmmBuffer", _make_buffer)
    plan = ekc.plan_extensible_kv_cache(
        ekc.ExtensibleKVCacheConfig(
            bytes_per_block=bytes_per_block,
            max_num_blocks=max_blocks,
            device_index=0,
            transient_peak_bytes=0,
        )
    )
    assert plan is not None
    return plan, created["buffer"]


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert ekc.plan_extensible_kv_cache(
        ekc.ExtensibleKVCacheConfig(1 << 20, 10, 0)
    ) is None


def test_plan_returns_none_when_vmm_unavailable(enabled):
    enabled.setattr(ekc, "vmm_unavailable_reason", lambda: "no driver")
    assert ekc.plan_extensible_kv_cache(
        ekc.ExtensibleKVCacheConfig(1 << 20, 10, 0)
    ) is None


def test_final_num_blocks_respects_margin_and_cap(enabled, monkeypatch):
    plan, _ = _plan(monkeypatch, max_blocks=100, bytes_per_block=1 << 20)
    # 100 MiB free, 1 GiB floor margin -> no usable memory.
    assert plan.final_num_blocks(100 << 20) == 0
    # 4 GiB free -> (4 GiB - 1 GiB) / 1 MiB = 3072, capped at 100.
    assert plan.final_num_blocks(4 << 30) == 100


def test_commit_is_idempotent_and_monotonic(enabled, monkeypatch):
    plan, buf = _plan(monkeypatch, max_blocks=100, bytes_per_block=1 << 20)
    assert plan.commit(10) == 10
    assert plan.commit(5) == 10
    assert plan.commit(20) == 20
    assert buf.committed_bytes == 20 << 20


def test_commit_extensible_uses_measured_free_memory(enabled, monkeypatch):
    plan, _ = _plan(monkeypatch, max_blocks=100, bytes_per_block=1 << 20)
    committed = ekc.commit_extensible_kv_cache(plan, free_bytes=4 << 30)
    assert committed == 100
    assert plan.committed_num_blocks() == 100


def test_fallback_num_blocks_matches_estimate_path():
    assert ekc.fallback_num_blocks(1 << 20, 4 << 30, 100) == 100
    assert ekc.fallback_num_blocks(1 << 20, 10 << 20, 100) == 10
    with pytest.raises(ValueError):
        ekc.fallback_num_blocks(0, 1, 1)


def test_all_ranks_support_returns_local_without_group():
    assert ekc.all_ranks_support_extensible_kv_cache(True) is True
    assert ekc.all_ranks_support_extensible_kv_cache(False) is False


def test_all_ranks_support_uses_min_reduce(monkeypatch):
    class _Group:
        def __init__(self) -> None:
            self.calls = []

        def all_reduce(self, tensor, op=None):
            self.calls.append(op)
            tensor.fill_(0)

    group = _Group()
    assert ekc.all_ranks_support_extensible_kv_cache(True, group) is False
    assert group.calls
