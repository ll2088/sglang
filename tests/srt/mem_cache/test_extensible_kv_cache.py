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


def test_plan_returns_none_for_invalid_config(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=0, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(cfg) is None


def test_final_num_blocks_respects_margin_and_cap():
    class _FakeBuffer:
        reserved_size = 64 << 20

        def commit(self, n):
            self.committed = n

        def free(self):
            pass

    plan = ekc.ExtensibleKVCachePlan(
        buffer=_FakeBuffer(),
        bytes_per_block=1 << 20,
        max_num_blocks=64,
        transient_peak_bytes=0,
    )
    # Free memory below the floor margin yields zero blocks.
    assert plan.final_num_blocks(0) == 0
    # Plenty of free memory is capped by max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 64
    # A measured transient peak shrinks the usable budget.
    plan.transient_peak_bytes = 8 << 20
    blocks = plan.final_num_blocks(32 << 20)
    assert 0 <= blocks < 32


def test_final_num_blocks_rejects_negative():
    class _FakeBuffer:
        reserved_size = 1 << 20

        def commit(self, n):
            pass

        def free(self):
            pass

    plan = ekc.ExtensibleKVCachePlan(
        buffer=_FakeBuffer(), bytes_per_block=1 << 20, max_num_blocks=1
    )
    with pytest.raises(ValueError):
        plan.final_num_blocks(-1)


@requires_cuda
def test_commit_is_idempotent(monkeypatch):
    _require_vmm()
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=2 << 20,
        max_num_blocks=4,
        device_index=torch.cuda.current_device(),
    )
    plan = plan_extensible_kv_cache(cfg)
    if plan is None:
        pytest.skip("reservation unavailable")
    try:
        first = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
        second = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
        assert first == second == 4
        # Committing a smaller count afterwards is a no-op.
        assert plan.commit(1) == 4
    finally:
        plan.free()


@requires_cuda
def test_commit_never_exceeds_reservation(monkeypatch):
    _require_vmm()
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=2 << 20,
        max_num_blocks=2,
        device_index=torch.cuda.current_device(),
    )
    plan = plan_extensible_kv_cache(cfg)
    if plan is None:
        pytest.skip("reservation unavailable")
    try:
        blocks = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
        assert blocks <= cfg.max_num_blocks
        assert plan.buffer.committed_bytes <= plan.buffer.reserved_size
    finally:
        plan.free()


def test_fallback_when_vmm_unavailable(monkeypatch):
    """When VMM is unavailable the planner returns None so callers fall back."""
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: "forced")
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(cfg) is None
