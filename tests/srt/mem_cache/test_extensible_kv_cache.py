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

from sglang.srt.mem_cache import extensible_kv_cache as ekc
from sglang.srt.mem_cache.extensible_kv_cache import (
    ExtensibleKVCacheConfig,
    ExtensibleKVCachePlan,
    commit_extensible_kv_cache,
    extensible_kv_cache_enabled,
    plan_extensible_kv_cache,
)
from sglang.srt.mem_cache.vmm_allocator import VmmBuffer, vmm_unavailable_reason

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
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(cfg) is None


def test_final_num_blocks_respects_cap_and_margin():
    plan = ExtensibleKVCachePlan(
        buffer=None,  # type: ignore[arg-type]
        bytes_per_block=1 << 20,
        max_num_blocks=8,
        transient_peak_bytes=0,
    )
    # Free memory below the floor leaves zero usable blocks.
    assert plan.final_num_blocks(0) == 0
    # Large free memory is capped by max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 8


def test_final_num_blocks_is_monotonic():
    plan = ExtensibleKVCachePlan(
        buffer=None,  # type: ignore[arg-type]
        bytes_per_block=1 << 20,
        max_num_blocks=64,
        transient_peak_bytes=0,
    )
    prev = -1
    for free in (0, 1 << 30, 2 << 30, 4 << 30, 8 << 30):
        blocks = plan.final_num_blocks(free)
        assert blocks >= prev
        prev = blocks


@requires_cuda
def test_commit_matches_measured_free_memory(monkeypatch):
    _require_vmm()
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    device_index = torch.cuda.current_device()
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=2 << 20,
        max_num_blocks=4,
        device_index=device_index,
    )
    plan = plan_extensible_kv_cache(cfg)
    assert plan is not None
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
        expected = plan.final_num_blocks(free_bytes)
        committed = commit_extensible_kv_cache(plan, free_bytes=free_bytes)
        assert committed == expected
        # Idempotent: committing again yields the same count.
        assert commit_extensible_kv_cache(plan, free_bytes=free_bytes) == expected
        assert plan.buffer.committed_bytes <= plan.buffer.reserved_size
    finally:
        plan.free()


@requires_cuda
def test_commit_never_exceeds_reservation(monkeypatch):
    _require_vmm()
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    device_index = torch.cuda.current_device()
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=2 << 20,
        max_num_blocks=2,
        device_index=device_index,
    )
    plan = plan_extensible_kv_cache(cfg)
    assert plan is not None
    try:
        committed = plan.commit(plan.max_num_blocks)
        assert committed <= plan.max_num_blocks
        assert plan.buffer.committed_bytes <= plan.buffer.reserved_size
    finally:
        plan.free()


def test_plan_falls_back_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: "forced")
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(cfg) is None


def test_plan_falls_back_on_reservation_error(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: None)

    def _boom(*args, **kwargs):
        raise ekc.VmmUnavailableError("boom")

    monkeypatch.setattr(ekc, "VmmBuffer", _boom)
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(cfg) is None
