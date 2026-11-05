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


class _FakeBuffer:
    """Stand-in for VmmBuffer that records commits without touching a driver."""

    def __init__(self, reserved_size: int) -> None:
        self.reserved_size = reserved_size
        self.committed_bytes = 0
        self.freed = False

    def commit(self, num_bytes: int) -> int:
        if self.freed:
            raise RuntimeError("freed")
        if num_bytes < 0:
            raise ValueError("negative")
        if num_bytes > self.reserved_size:
            raise ValueError("exceeds reservation")
        self.committed_bytes = max(self.committed_bytes, num_bytes)
        return self.committed_bytes

    def free(self) -> None:
        self.freed = True


def _make_plan(bytes_per_block: int = 1024, max_num_blocks: int = 100):
    buffer = _FakeBuffer(bytes_per_block * max_num_blocks)
    return ekc.ExtensibleKVCachePlan(
        buffer=buffer,
        bytes_per_block=bytes_per_block,
        max_num_blocks=max_num_blocks,
        transient_peak_bytes=0,
    )


def test_final_num_blocks_respects_cap_and_margin():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    # Free memory far above the reservation: capped at max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 100
    # Free memory below the floor margin: no blocks.
    assert plan.final_num_blocks(0) == 0
    # Free memory just above the floor margin: a small positive count.
    blocks = plan.final_num_blocks(ekc.KV_CACHE_MARGIN_FLOOR_BYTES + 10 * 1024)
    assert 0 < blocks <= 100


def test_commit_is_idempotent_and_monotonic():
    plan = _make_plan()
    first = plan.commit(10)
    assert first == 10
    assert plan.commit(10) == 10
    # Committing fewer blocks after more is a no-op.
    assert plan.commit(3) == 10
    assert plan.buffer.committed_bytes == 10 * plan.bytes_per_block


def test_commit_rejects_negative_blocks():
    plan = _make_plan()
    with pytest.raises(ValueError):
        plan.commit(-1)


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "0")
    config = ekc.ExtensibleKVCacheConfig(
        bytes_per_block=1024, max_num_blocks=10, device_index=0
    )
    assert ekc.plan_extensible_kv_cache(config) is None


def test_plan_returns_none_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: "no driver")
    config = ekc.ExtensibleKVCacheConfig(
        bytes_per_block=1024, max_num_blocks=10, device_index=0
    )
    assert ekc.plan_extensible_kv_cache(config) is None


def test_plan_returns_none_on_reservation_failure(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: None)

    def _boom(*args, **kwargs):
        raise ekc.VmmUnavailableError("reserve failed")

    monkeypatch.setattr(ekc, "VmmBuffer", _boom)
    config = ekc.ExtensibleKVCacheConfig(
        bytes_per_block=1024, max_num_blocks=10, device_index=0
    )
    assert ekc.plan_extensible_kv_cache(config) is None


def test_commit_extensible_kv_cache_uses_supplied_free_bytes():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    committed = ekc.commit_extensible_kv_cache(
        plan, free_bytes=ekc.KV_CACHE_MARGIN_FLOOR_BYTES + 20 * 1024
    )
    assert committed == 20
    assert plan.buffer.committed_bytes == 20 * plan.bytes_per_block
