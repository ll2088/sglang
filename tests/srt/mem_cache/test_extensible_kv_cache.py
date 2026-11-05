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

A fake ``VmmBuffer`` stands in for the driver-backed reservation so the
sizing, idempotency and fallback logic can be checked without a GPU.
"""

from __future__ import annotations

import pytest

from sglang.srt.mem_cache import extensible_kv_cache as ekc


class _FakeBuffer:
    def __init__(self, reserved_size: int, device_index: int = 0):
        self.reserved_size = reserved_size
        self.device_index = device_index
        self._committed = 0
        self.freed = False

    @property
    def committed_bytes(self) -> int:
        return self._committed

    def commit(self, num_bytes: int) -> int:
        if num_bytes > self.reserved_size:
            raise ValueError("exceeds reservation")
        self._committed = max(self._committed, num_bytes)
        return self._committed

    def free(self) -> None:
        self.freed = True
        self._committed = 0


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: None)
    return monkeypatch


def _plan(monkeypatch, max_blocks=100, bytes_per_block=1 << 20, transient=0):
    monkeypatch.setattr(
        ekc,
        "VmmBuffer",
        lambda size, device: _FakeBuffer(size, device),
    )
    return ekc.plan_extensible_kv_cache(
        ekc.ExtensibleKVCacheConfig(
            bytes_per_block=bytes_per_block,
            max_num_blocks=max_blocks,
            device_index=0,
            transient_peak_bytes=transient,
        )
    )


def test_disabled_returns_none(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert ekc.plan_extensible_kv_cache(
        ekc.ExtensibleKVCacheConfig(1 << 20, 10, 0)
    ) is None


def test_unavailable_vmm_returns_none(enabled):
    enabled.setattr(ekc, "vmm_unavailable_reason", lambda: "no driver")
    assert ekc.plan_extensible_kv_cache(
        ekc.ExtensibleKVCacheConfig(1 << 20, 10, 0)
    ) is None


def test_final_blocks_respect_margin_and_cap(enabled, monkeypatch):
    plan = _plan(monkeypatch, max_blocks=100, bytes_per_block=1 << 20)
    assert plan is not None
    # 100 MiB free, 1 GiB floor margin -> nothing usable.
    assert plan.final_num_blocks(100 << 20) == 0
    # 2 GiB free -> (2 GiB - 1 GiB) / 1 MiB = 1024, capped at 100.
    assert plan.final_num_blocks(2 << 30) == 100


def test_commit_is_idempotent_and_monotonic(enabled, monkeypatch):
    plan = _plan(monkeypatch, max_blocks=10, bytes_per_block=1 << 20)
    assert plan is not None
    assert plan.commit(3) == 3
    assert plan.commit(3) == 3
    assert plan.commit(5) == 5
    assert plan.commit(2) == 5
    assert plan.committed_num_blocks() == 5


def test_commit_extensible_uses_measured_free(enabled, monkeypatch):
    plan = _plan(monkeypatch, max_blocks=100, bytes_per_block=1 << 20)
    assert plan is not None
    committed = ekc.commit_extensible_kv_cache(plan, free_bytes=2 << 30)
    assert committed == 100


def test_fallback_matches_estimate_path():
    assert ekc.fallback_num_blocks(1 << 20, 2 << 30, 100) == 100
    assert ekc.fallback_num_blocks(1 << 20, 100 << 20, 100) == 100
    assert ekc.fallback_num_blocks(1 << 20, 10 << 20, 100) == 10


def test_all_ranks_support_requires_unanimity():
    assert ekc.all_ranks_support_extensible_kv_cache(True) is True
    assert ekc.all_ranks_support_extensible_kv_cache(False) is False


def test_plan_free_releases_buffer(enabled, monkeypatch):
    plan = _plan(monkeypatch)
    assert plan is not None
    plan.commit(1)
    plan.free()
    assert plan.buffer.freed
