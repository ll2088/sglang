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
"""Two-phase sizing and fallback behaviour for the extensible KV cache.

These tests exercise the pure sizing logic without requiring a GPU: the VMM
reservation is replaced by a fake buffer so the measurement, idempotency and
fallback paths can run on CPU-only CI hosts.
"""

import pytest

from sglang.srt.mem_cache import extensible_kv_cache as ekc


class _FakeBuffer:
    """Stand-in for VmmBuffer that records commits without touching a driver."""

    def __init__(self, max_bytes, device_index):
        self.reserved_size = max_bytes
        self.device_index = device_index
        self.committed = 0
        self.freed = False

    def commit(self, num_bytes):
        if self.freed:
            raise RuntimeError("freed")
        if num_bytes > self.reserved_size:
            raise ValueError("exceeds reservation")
        self.committed = max(self.committed, num_bytes)
        return self.committed

    def free(self):
        self.freed = True


@pytest.fixture
def fake_vmm(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: None)
    monkeypatch.setattr(ekc, "VmmBuffer", _FakeBuffer)
    return monkeypatch


def _config(bytes_per_block=1024, max_num_blocks=100, transient=0):
    return ekc.ExtensibleKVCacheConfig(
        bytes_per_block=bytes_per_block,
        max_num_blocks=max_num_blocks,
        device_index=0,
        transient_peak_bytes=transient,
    )


def test_disabled_returns_none(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert ekc.plan_extensible_kv_cache(_config()) is None


def test_unavailable_vmm_returns_none(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: "no driver")
    assert ekc.plan_extensible_kv_cache(_config()) is None


def test_final_blocks_respect_margin_and_cap(fake_vmm):
    plan = ekc.plan_extensible_kv_cache(_config())
    assert plan is not None
    # Free memory far above the cap: the reservation is the upper bound.
    assert plan.final_num_blocks(1 << 40) == 100
    # Free memory below the floor: nothing is committed.
    assert plan.final_num_blocks(0) == 0


def test_commit_is_idempotent_and_monotonic(fake_vmm):
    plan = ekc.plan_extensible_kv_cache(_config())
    assert plan is not None
    first = plan.commit(10)
    second = plan.commit(10)
    assert first == second == 10
    # Committing fewer blocks after more is a no-op.
    assert plan.commit(5) == 10
    # Committing more grows the reservation.
    assert plan.commit(20) == 20


def test_commit_never_exceeds_reservation(fake_vmm):
    plan = ekc.plan_extensible_kv_cache(_config())
    assert plan is not None
    assert plan.commit(1000) == 100


def test_commit_after_free_raises(fake_vmm):
    plan = ekc.plan_extensible_kv_cache(_config())
    assert plan is not None
    plan.free()
    with pytest.raises(RuntimeError):
        plan.commit(1)


def test_commit_uses_measured_free_memory(fake_vmm):
    plan = ekc.plan_extensible_kv_cache(_config())
    assert plan is not None
    # 64 KiB free, 1 GiB floor: nothing usable, so no blocks are committed.
    assert ekc.commit_extensible_kv_cache(plan, free_bytes=64 << 10) == 0
    # 2 GiB free: floor is 1 GiB, so ~1 GiB / 1 KiB blocks are usable.
    committed = ekc.commit_extensible_kv_cache(plan, free_bytes=2 << 30)
    assert committed == 100


def test_fallback_equivalence(fake_vmm):
    """When VMM is unavailable the caller keeps the estimate path; the plan
    helper must not raise and must return None."""
    fake_vmm.setattr(ekc, "vmm_unavailable_reason", lambda: "unsupported")
    assert ekc.plan_extensible_kv_cache(_config()) is None
