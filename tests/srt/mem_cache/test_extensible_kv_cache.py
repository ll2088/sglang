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
import torch

from sglang.srt.mem_cache.extensible_kv_cache import (
    ExtensibleKVCacheConfig,
    ExtensibleKVCachePlan,
    commit_extensible_kv_cache,
    extensible_kv_cache_enabled,
    plan_extensible_kv_cache,
)
from sglang.srt.mem_cache.vmm_allocator import (
    VmmBuffer,
    vmm_unavailable_reason,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _require_vmm():
    reason = vmm_unavailable_reason()
    if reason is not None:
        pytest.skip(f"VMM unavailable: {reason}")


def test_enabled_defaults_to_false(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False


def test_enabled_reads_env(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_final_num_blocks_respects_cap_and_margin():
    # Build a plan without touching the driver: only the pure sizing logic is
    # exercised here.
    plan = ExtensibleKVCachePlan.__new__(ExtensibleKVCachePlan)
    plan.buffer = None
    plan.bytes_per_block = 1 << 20
    plan.max_num_blocks = 100
    plan.transient_peak_bytes = 0

    # Free memory below the floor margin yields zero blocks.
    assert plan.final_num_blocks(0) == 0
    # Plenty of free memory is capped at max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 100
    # A moderate amount yields a positive count below the cap.
    blocks = plan.final_num_blocks(8 << 30)
    assert 0 < blocks <= 100


def test_final_num_blocks_rejects_negative():
    plan = ExtensibleKVCachePlan.__new__(ExtensibleKVCachePlan)
    plan.buffer = None
    plan.bytes_per_block = 1 << 20
    plan.max_num_blocks = 10
    plan.transient_peak_bytes = 0
    with pytest.raises(ValueError):
        plan.final_num_blocks(-1)


@requires_cuda
def test_plan_and_commit_roundtrip(monkeypatch):
    _require_vmm()
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    device_index = torch.cuda.current_device()
    config = ExtensibleKVCacheConfig(
        bytes_per_block=2 << 20,
        max_num_blocks=4,
        device_index=device_index,
    )
    plan = plan_extensible_kv_cache(config)
    assert plan is not None
    try:
        # Commit a small count and verify it is idempotent.
        first = plan.commit(1)
        second = plan.commit(1)
        assert first == second
        assert first <= config.max_num_blocks
        # Committing fewer blocks after more is a no-op.
        assert plan.commit(0) == first
    finally:
        plan.free()


@requires_cuda
def test_commit_uses_measured_free_memory(monkeypatch):
    _require_vmm()
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    device_index = torch.cuda.current_device()
    config = ExtensibleKVCacheConfig(
        bytes_per_block=2 << 20,
        max_num_blocks=4,
        device_index=device_index,
    )
    plan = plan_extensible_kv_cache(config)
    assert plan is not None
    try:
        # Supplying free_bytes explicitly makes the result deterministic.
        committed = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
        assert committed == config.max_num_blocks
        # Repeated calls are idempotent.
        assert commit_extensible_kv_cache(plan, free_bytes=1 << 40) == committed
    finally:
        plan.free()


def test_plan_falls_back_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(
        "sglang.srt.mem_cache.extensible_kv_cache.vmm_unavailable_reason",
        lambda: "forced unavailable",
    )
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_falls_back_on_reservation_error(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(
        "sglang.srt.mem_cache.extensible_kv_cache.vmm_unavailable_reason",
        lambda: None,
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("reservation failed")

    monkeypatch.setattr(
        "sglang.srt.mem_cache.extensible_kv_cache.VmmBuffer", _boom
    )
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None
