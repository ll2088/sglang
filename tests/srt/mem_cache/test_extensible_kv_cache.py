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

from sglang.srt.mem_cache import extensible_kv_cache as ekv
from sglang.srt.mem_cache.extensible_kv_cache import (
    ExtensibleKVCacheConfig,
    ExtensibleKVCachePlan,
    commit_extensible_kv_cache,
    extensible_kv_cache_enabled,
    plan_extensible_kv_cache,
)
from sglang.srt.mem_cache.vmm_allocator import (
    VmmBuffer,
    VmmUnavailableError,
    vmm_unavailable_reason,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _require_vmm():
    reason = vmm_unavailable_reason()
    if reason is not None:
        pytest.skip(f"VMM unavailable: {reason}")


def test_enabled_flag_reads_env(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_final_num_blocks_respects_margin_and_cap():
    plan = ExtensibleKVCachePlan(
        buffer=None,  # type: ignore[arg-type]
        bytes_per_block=1 << 20,
        max_num_blocks=100,
        transient_peak_bytes=0,
    )
    # Free memory below the floor margin yields zero blocks.
    assert plan.final_num_blocks(0) == 0
    # Large free memory is capped by max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 100
    # A measured transient peak scales the margin.
    plan.transient_peak_bytes = 4 << 30
    margin = max(
        ekv.KV_CACHE_MARGIN_FLOOR_BYTES,
        int(plan.transient_peak_bytes * ekv.KV_CACHE_MARGIN_FRACTION),
    )
    free = margin + 10 * (1 << 20)
    assert plan.final_num_blocks(free) == 10


def test_final_num_blocks_rejects_negative_free():
    plan = ExtensibleKVCachePlan(
        buffer=None,  # type: ignore[arg-type]
        bytes_per_block=1 << 20,
        max_num_blocks=8,
    )
    with pytest.raises(ValueError):
        plan.final_num_blocks(-1)


@requires_cuda
def test_plan_and_commit_roundtrip(monkeypatch):
    _require_vmm()
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=2 << 20,
        max_num_blocks=4,
        device_index=torch.cuda.current_device(),
    )
    plan = plan_extensible_kv_cache(config)
    assert plan is not None
    try:
        committed = plan.commit(2)
        assert committed == 2
        # Idempotent: committing fewer blocks is a no-op.
        assert plan.commit(1) == 2
        # Committing more than the cap is clamped.
        assert plan.commit(100) == 4
    finally:
        plan.free()


@requires_cuda
def test_commit_measures_free_memory(monkeypatch):
    _require_vmm()
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=2 << 20,
        max_num_blocks=4,
        device_index=torch.cuda.current_device(),
    )
    plan = plan_extensible_kv_cache(config)
    assert plan is not None
    try:
        blocks = commit_extensible_kv_cache(plan, free_bytes=0)
        assert blocks == 0
        blocks = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
        assert blocks == 4
    finally:
        plan.free()


def test_plan_falls_back_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekv, "vmm_unavailable_reason", lambda: "forced")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_falls_back_on_reservation_error(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekv, "vmm_unavailable_reason", lambda: None)

    def _boom(*args, **kwargs):
        raise VmmUnavailableError("boom")

    monkeypatch.setattr(ekv, "VmmBuffer", _boom)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_rejects_invalid_config(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekv, "vmm_unavailable_reason", lambda: None)
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
