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
    commit_extensible_kv_cache,
    extensible_kv_cache_enabled,
    plan_extensible_kv_cache,
)
from sglang.srt.mem_cache.vmm_allocator import vmm_unavailable_reason

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _require_vmm():
    reason = vmm_unavailable_reason()
    if reason is not None:
        pytest.skip(f"VMM unavailable: {reason}")


def test_enabled_reads_env(monkeypatch):
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


def test_plan_returns_none_for_invalid_config(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=0, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_final_num_blocks_respects_margin_and_cap():
    # Build a plan-like object without touching the driver.
    class _FakeBuffer:
        reserved_size = 1 << 30
        device_index = 0

        def commit(self, n):
            self.committed = n

        def free(self):
            pass

    plan = ekc.ExtensibleKVCachePlan(
        buffer=_FakeBuffer(),
        bytes_per_block=1 << 20,
        max_num_blocks=1024,
        transient_peak_bytes=0,
    )
    # Free memory below the floor yields zero blocks.
    assert plan.final_num_blocks(0) == 0
    # Free memory above the floor is capped by max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 1024
    # A measured transient peak raises the margin.
    plan.transient_peak_bytes = 2 << 30
    assert plan.final_num_blocks(3 << 30) == 0


def test_commit_is_idempotent():
    class _FakeBuffer:
        reserved_size = 1 << 30
        device_index = 0

        def __init__(self):
            self.committed = 0

        def commit(self, n):
            self.committed = max(self.committed, n)

        def free(self):
            pass

    plan = ekc.ExtensibleKVCachePlan(
        buffer=_FakeBuffer(),
        bytes_per_block=1 << 20,
        max_num_blocks=16,
    )
    assert plan.commit(4) == 4
    assert plan.commit(4) == 4
    assert plan.commit(2) == 4


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
        free_bytes, _ = torch.cuda.mem_get_info(plan.buffer.device_index)
        blocks = commit_extensible_kv_cache(plan, free_bytes=free_bytes)
        assert 0 <= blocks <= config.max_num_blocks
        # Repeated commits are idempotent.
        assert commit_extensible_kv_cache(plan, free_bytes=free_bytes) == blocks
    finally:
        plan.free()


@requires_cuda
def test_commit_failure_falls_back(monkeypatch):
    _require_vmm()
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=2 << 20,
        max_num_blocks=2,
        device_index=torch.cuda.current_device(),
    )
    plan = plan_extensible_kv_cache(config)
    assert plan is not None
    try:
        # A negative free-memory measurement must not commit anything and
        # must surface as a ValueError so callers can fall back.
        with pytest.raises(ValueError):
            commit_extensible_kv_cache(plan, free_bytes=-1)
        assert plan.buffer.committed_bytes == 0
    finally:
        plan.free()
