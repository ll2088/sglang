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
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_returns_none_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekv, "vmm_unavailable_reason", lambda: "forced")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_returns_none_on_reservation_failure(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekv, "vmm_unavailable_reason", lambda: None)

    def _boom(*args, **kwargs):
        raise VmmUnavailableError("forced reservation failure")

    monkeypatch.setattr(ekv, "VmmBuffer", _boom)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_final_num_blocks_respects_cap_and_margin():
    plan = ExtensibleKVCachePlan.__new__(ExtensibleKVCachePlan)
    plan.buffer = None
    plan.bytes_per_block = 1 << 20
    plan.max_num_blocks = 100
    # Free memory far above the cap: never exceed max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 100
    # Free memory below the floor margin: no blocks.
    assert plan.final_num_blocks(0) == 0


def test_commit_is_idempotent_and_monotonic():
    plan = ExtensibleKVCachePlan.__new__(ExtensibleKVCachePlan)
    plan.bytes_per_block = 1 << 20
    plan.max_num_blocks = 8
    plan.buffer = _FakeBuffer()
    first = plan.commit(4)
    second = plan.commit(4)
    assert first == second == 4
    smaller = plan.commit(1)
    assert smaller == 4
    assert plan.buffer.committed_bytes == 4 << 20


class _FakeBuffer:
    def __init__(self):
        self.committed_bytes = 0

    def commit(self, num_bytes):
        self.committed_bytes = max(self.committed_bytes, num_bytes)
        return self.committed_bytes

    def free(self):
        self.committed_bytes = 0


@requires_cuda
def test_commit_uses_measured_free_memory(monkeypatch):
    _require_vmm()
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    device_index = torch.cuda.current_device()
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=device_index
    )
    plan = plan_extensible_kv_cache(config)
    if plan is None:
        pytest.skip("VMM reservation unavailable on this host")
    try:
        free_bytes = 64 << 20
        committed = commit_extensible_kv_cache(plan, free_bytes=free_bytes)
        assert 0 <= committed <= plan.max_num_blocks
        assert plan.buffer.committed_bytes <= plan.reserved_bytes
        # Repeated calls are idempotent.
        assert commit_extensible_kv_cache(plan, free_bytes=free_bytes) == committed
    finally:
        plan.free()
