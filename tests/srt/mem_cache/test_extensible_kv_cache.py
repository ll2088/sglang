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

"""Unit tests for the two-phase extensible KV cache sizing helpers.

These tests exercise the pure sizing/commit logic without requiring a GPU:
the reservation phase is replaced by a fake buffer so the tests run in CI
where VMM is unavailable, and the fallback path is verified explicitly.
"""

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
    """Stand-in for VmmBuffer that records commits without touching a driver."""

    def __init__(self, reserved_size: int, device_index: int = 0) -> None:
        self.reserved_size = reserved_size
        self.device_index = device_index
        self.committed_bytes = 0
        self.freed = False

    def commit(self, num_bytes: int) -> int:
        if self.freed:
            raise RuntimeError("VmmBuffer has been freed")
        if num_bytes < 0:
            raise ValueError("num_bytes must be non-negative")
        if num_bytes > self.reserved_size:
            raise ValueError("commit exceeds reservation")
        # Monotonic and idempotent, matching VmmBuffer semantics.
        self.committed_bytes = max(self.committed_bytes, num_bytes)
        return self.committed_bytes

    def free(self) -> None:
        self.freed = True


def _make_plan(
    bytes_per_block: int = 1 << 20,
    max_num_blocks: int = 64,
    transient_peak_bytes: int = 0,
) -> ExtensibleKVCachePlan:
    return ExtensibleKVCachePlan(
        buffer=_FakeBuffer(bytes_per_block * max_num_blocks),
        bytes_per_block=bytes_per_block,
        max_num_blocks=max_num_blocks,
        transient_peak_bytes=transient_peak_bytes,
    )


def test_enabled_defaults_to_false(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False


def test_enabled_by_env(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_final_num_blocks_respects_cap():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=8)
    # Plenty of free memory: capped at max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 8


def test_final_num_blocks_keeps_margin_floor():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=64)
    free = ekc.KV_CACHE_MARGIN_FLOOR_BYTES + (4 << 20)
    assert plan.final_num_blocks(free) == 4


def test_final_num_blocks_uses_measured_transient():
    plan = _make_plan(
        bytes_per_block=1 << 20, max_num_blocks=64, transient_peak_bytes=8 << 20
    )
    free = (8 << 20) + (4 << 20) + (8 << 20)
    # margin = max(floor, 0.75 * transient) = 6 MiB, usable = 14 MiB.
    assert plan.final_num_blocks(free) == 14


def test_final_num_blocks_never_negative():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=8)
    assert plan.final_num_blocks(0) == 0
    with pytest.raises(ValueError):
        plan.final_num_blocks(-1)


def test_commit_is_idempotent_and_monotonic():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=8)
    assert plan.commit(3) == 3
    assert plan.commit(3) == 3
    assert plan.commit(1) == 3
    assert plan.commit(8) == 8


def test_commit_clamps_to_max_blocks():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=4)
    assert plan.commit(100) == 4


def test_commit_rejects_negative():
    plan = _make_plan()
    with pytest.raises(ValueError):
        plan.commit(-1)


def test_commit_after_free_raises():
    plan = _make_plan()
    plan.free()
    with pytest.raises(RuntimeError):
        plan.commit(1)


def test_commit_extensible_kv_cache_uses_supplied_free_bytes():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=64)
    free = ekc.KV_CACHE_MARGIN_FLOOR_BYTES + (5 << 20)
    assert commit_extensible_kv_cache(plan, free_bytes=free) == 5


def test_commit_extensible_kv_cache_is_repeatable():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=64)
    free = ekc.KV_CACHE_MARGIN_FLOOR_BYTES + (5 << 20)
    first = commit_extensible_kv_cache(plan, free_bytes=free)
    second = commit_extensible_kv_cache(plan, free_bytes=free)
    assert first == second == 5


def test_plan_falls_back_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: "no driver")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_falls_back_on_reservation_error(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: None)

    def _boom(*args, **kwargs):
        raise ekc.VmmUnavailableError("reserve failed")

    monkeypatch.setattr(ekc, "VmmBuffer", _boom)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_rejects_invalid_config(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: None)
    assert (
        plan_extensible_kv_cache(
            ExtensibleKVCacheConfig(
                bytes_per_block=0, max_num_blocks=8, device_index=0
            )
        )
        is None
    )
    assert (
        plan_extensible_kv_cache(
            ExtensibleKVCacheConfig(
                bytes_per_block=1 << 20, max_num_blocks=0, device_index=0
            )
        )
        is None
    )
