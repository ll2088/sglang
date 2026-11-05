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

"""Two-phase sizing helpers: measurement, idempotency and fallback.

These tests exercise the pure sizing logic without requiring a GPU. The
VMM-backed commit path is covered separately in ``test_vmm_allocator.py``.
"""

import pytest

from sglang.srt.mem_cache import extensible_kv_cache as ekc


class _FakeBuffer:
    def __init__(self, reserved_size: int) -> None:
        self.reserved_size = reserved_size
        self.committed = 0
        self.freed = False

    def commit(self, num_bytes: int) -> int:
        assert 0 <= num_bytes <= self.reserved_size
        self.committed = max(self.committed, num_bytes)
        return self.committed

    def free(self) -> None:
        self.freed = True


@pytest.fixture
def plan():
    buffer = _FakeBuffer(reserved_size=64 << 20)
    return ekc.ExtensibleKVCachePlan(
        buffer=buffer,
        bytes_per_block=1 << 20,
        max_num_blocks=64,
        transient_peak_bytes=4 << 20,
    )


def test_final_num_blocks_uses_measured_free_memory(plan):
    # 32 MiB free, margin = max(1 GiB floor, 0.75 * 4 MiB) = 1 GiB, so the
    # usable budget is zero and no blocks are committed.
    assert plan.final_num_blocks(32 << 20) == 0


def test_final_num_blocks_never_exceeds_reservation(plan):
    assert plan.final_num_blocks(1 << 40) == plan.max_num_blocks


def test_final_num_blocks_rejects_negative(plan):
    with pytest.raises(ValueError):
        plan.final_num_blocks(-1)


def test_commit_is_idempotent_and_monotonic(plan):
    first = plan.commit(8)
    second = plan.commit(8)
    assert first == second == 8
    smaller = plan.commit(2)
    assert smaller == first


def test_commit_clamps_to_max(plan):
    assert plan.commit(10_000) == plan.max_num_blocks


def test_free_releases_buffer(plan):
    plan.free()
    assert plan.buffer.freed is True


def test_plan_returns_none_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: "no driver")
    config = ekc.ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert ekc.plan_extensible_kv_cache(config) is None


def test_plan_returns_none_for_invalid_config(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: None)
    config = ekc.ExtensibleKVCacheConfig(
        bytes_per_block=0, max_num_blocks=4, device_index=0
    )
    assert ekc.plan_extensible_kv_cache(config) is None


def test_commit_measures_free_memory_when_not_supplied(monkeypatch, plan):
    monkeypatch.setattr(
        ekc.torch.cuda, "mem_get_info", lambda _idx: (1 << 40, 1 << 40)
    )
    assert ekc.commit_extensible_kv_cache(plan) == plan.max_num_blocks
