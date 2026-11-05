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
from sglang.srt.mem_cache.vmm_allocator import vmm_unavailable_reason

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _require_vmm():
    reason = vmm_unavailable_reason()
    if reason is not None:
        pytest.skip(f"VMM unavailable: {reason}")


def test_feature_is_opt_in(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert not extensible_kv_cache_enabled()
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled()


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_final_num_blocks_respects_cap_and_margin():
    plan = ExtensibleKVCachePlan.__new__(ExtensibleKVCachePlan)
    plan.bytes_per_block = 1 << 20
    plan.max_num_blocks = 100
    # Free memory below the floor leaves no usable blocks.
    assert plan.final_num_blocks(0) == 0
    # Large free memory is capped by max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 100


@requires_cuda
def test_plan_and_commit_roundtrip(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    _require_vmm()
    config = ExtensibleKVCacheConfig(
        bytes_per_block=2 << 20,
        max_num_blocks=4,
        device_index=torch.cuda.current_device(),
    )
    plan = plan_extensible_kv_cache(config)
    assert plan is not None
    try:
        free_bytes, _ = torch.cuda.mem_get_info(config.device_index)
        committed = commit_extensible_kv_cache(plan, free_bytes=free_bytes)
        assert 0 <= committed <= config.max_num_blocks
        # Repeated commit is idempotent.
        assert commit_extensible_kv_cache(plan, free_bytes=free_bytes) == committed
    finally:
        plan.free()


@requires_cuda
def test_commit_falls_back_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setenv("SGLANG_DISABLE_VMM", "1")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20,
        max_num_blocks=4,
        device_index=torch.cuda.current_device(),
    )
    assert plan_extensible_kv_cache(config) is None
