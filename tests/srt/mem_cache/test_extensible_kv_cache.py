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
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_returns_none_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: "forced")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_final_num_blocks_respects_margin_and_cap():
    plan = ExtensibleKVCachePlan(
        buffer=None,  # type: ignore[arg-type]
        bytes_per_block=1 << 20,
        max_num_blocks=100,
        transient_peak_bytes=0,
    )
    # Free memory below the floor leaves no usable blocks.
    assert plan.final_num_blocks(0) == 0
    # Plenty of free memory is capped by max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 100
    # A measured transient peak scales the margin.
    plan.transient_peak_bytes = 4 << 30
    margin = max(ekc.KV_CACHE_MARGIN_FLOOR_BYTES, int((4 << 30) * ekc.KV_CACHE_MARGIN_FRACTION))
    free = margin + (10 << 20)
    assert plan.final_num_blocks(free) == 10


def test_final_num_blocks_rejects_negative_free():
    plan = ExtensibleKVCachePlan(
        buffer=None,  # type: ignore[arg-type]
        bytes_per_block=1 << 20,
        max_num_blocks=4,
    )
    with pytest.raises(ValueError):
        plan.final_num_blocks(-1)


@requires_cuda
def test_commit_is_idempotent_after_warmup():
    _require_vmm()
    device_index = torch.cuda.current_device()
    bytes_per_block = 2 << 20
    plan = ExtensibleKVCachePlan(
        buffer=VmmBuffer(bytes_per_block * 4, device_index),
        bytes_per_block=bytes_per_block,
        max_num_blocks=4,
    )
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
        first = commit_extensible_kv_cache(plan, free_bytes=free_bytes)
        second = commit_extensible_kv_cache(plan, free_bytes=free_bytes)
        assert first == second
        assert first <= plan.max_num_blocks
        assert plan.buffer.committed_bytes <= plan.buffer.reserved_size
    finally:
        plan.free()


@requires_cuda
def test_commit_never_exceeds_reservation():
    _require_vmm()
    device_index = torch.cuda.current_device()
    bytes_per_block = 2 << 20
    plan = ExtensibleKVCachePlan(
        buffer=VmmBuffer(bytes_per_block * 2, device_index),
        bytes_per_block=bytes_per_block,
        max_num_blocks=2,
    )
    try:
        committed = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
        assert committed == 2
        assert plan.buffer.committed_bytes == plan.buffer.reserved_size
    finally:
        plan.free()


def test_commit_failure_is_reported_for_fallback(monkeypatch):
    class _FailingBuffer:
        device_index = 0
        reserved_size = 1 << 20
        committed_bytes = 0

        def commit(self, num_bytes):
            raise VmmUnavailableError("forced mapping failure")

    plan = ExtensibleKVCachePlan(
        buffer=_FailingBuffer(),  # type: ignore[arg-type]
        bytes_per_block=1 << 20,
        max_num_blocks=1,
    )
    with pytest.raises(VmmUnavailableError):
        commit_extensible_kv_cache(plan, free_bytes=1 << 30)
