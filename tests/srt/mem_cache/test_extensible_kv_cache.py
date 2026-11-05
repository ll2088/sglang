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


def test_enabled_defaults_to_false(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False


def test_enabled_reads_env(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_returns_none_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: "forced")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_final_num_blocks_respects_cap_and_margin():
    plan = ExtensibleKVCachePlan.__new__(ExtensibleKVCachePlan)
    plan.bytes_per_block = 1 << 20
    plan.max_num_blocks = 100
    plan.transient_peak_bytes = 0
    # Free memory below the floor yields zero blocks, never negative.
    assert plan.final_num_blocks(0) == 0
    # A large free pool is capped by max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 100


def test_final_num_blocks_uses_transient_peak():
    plan = ExtensibleKVCachePlan.__new__(ExtensibleKVCachePlan)
    plan.bytes_per_block = 1 << 20
    plan.max_num_blocks = 1000
    plan.transient_peak_bytes = 4 << 30
    # Margin is 75% of the transient peak, i.e. 3 GiB.
    free = 8 << 30
    expected = (free - (3 << 30)) // (1 << 20)
    assert plan.final_num_blocks(free) == expected


def test_final_num_blocks_rejects_negative():
    plan = ExtensibleKVCachePlan.__new__(ExtensibleKVCachePlan)
    plan.bytes_per_block = 1 << 20
    plan.max_num_blocks = 10
    plan.transient_peak_bytes = 0
    with pytest.raises(ValueError):
        plan.final_num_blocks(-1)


@requires_cuda
def test_commit_is_idempotent_and_capped():
    _require_vmm()
    plan = plan_extensible_kv_cache(
        ExtensibleKVCacheConfig(
            bytes_per_block=2 << 20,
            max_num_blocks=4,
            device_index=torch.cuda.current_device(),
        )
    )
    assert plan is not None
    try:
        first = plan.commit(2)
        second = plan.commit(2)
        assert first == second
        # Committing more than the cap is clamped to max_num_blocks.
        assert plan.commit(100) == 4
        # Committing fewer afterwards is a no-op.
        assert plan.commit(1) == 4
    finally:
        plan.free()


@requires_cuda
def test_commit_measures_free_memory():
    _require_vmm()
    plan = plan_extensible_kv_cache(
        ExtensibleKVCacheConfig(
            bytes_per_block=2 << 20,
            max_num_blocks=4,
            device_index=torch.cuda.current_device(),
        )
    )
    assert plan is not None
    try:
        committed = commit_extensible_kv_cache(plan)
        assert 0 <= committed <= plan.max_num_blocks
        assert plan.buffer.committed_bytes == committed * plan.bytes_per_block
    finally:
        plan.free()


def test_commit_failure_is_reported():
    class _FailingBuffer:
        device_index = 0
        reserved_size = 1 << 20

        def commit(self, num_bytes):
            raise VmmUnavailableError("boom")

    plan = ExtensibleKVCachePlan(
        buffer=_FailingBuffer(),
        bytes_per_block=1 << 20,
        max_num_blocks=1,
    )
    with pytest.raises(VmmUnavailableError):
        plan.commit(1)


@requires_cuda
def test_fallback_equivalent_when_vmm_disabled(monkeypatch):
    """With VMM disabled the plan is None, so callers keep the old path."""
    monkeypatch.setenv("SGLANG_DISABLE_VMM", "1")
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20,
        max_num_blocks=4,
        device_index=torch.cuda.current_device(),
    )
    assert plan_extensible_kv_cache(config) is None


@requires_cuda
def test_buffer_free_releases_reservation():
    _require_vmm()
    buf = VmmBuffer(4 << 20, torch.cuda.current_device())
    buf.commit(2 << 20)
    buf.free()
    assert buf.base_ptr == 0
    with pytest.raises(RuntimeError):
        buf.commit(1 << 20)
