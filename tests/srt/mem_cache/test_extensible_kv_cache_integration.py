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
"""Integration checks for the two-phase KV cache sizing flow.

These tests exercise the plan/commit helpers the way model_runner and the
mem_cache layer use them: reserve an upper bound, measure free memory after
warmup, commit the final block count, and fall back to the estimate path when
VMM is unavailable. They do not require a GPU for the fallback cases.
"""

import pytest
import torch

from sglang.srt.mem_cache.extensible_kv_cache import (
    ExtensibleKVCacheConfig,
    ExtensibleKVCachePlan,
    commit_extensible_kv_cache,
    extensible_kv_cache_enabled,
    plan_extensible_kv_cache,
)
from sglang.srt.mem_cache.utils import extensible_kv_cache_supported
from sglang.srt.mem_cache.vmm_allocator import vmm_unavailable_reason

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _require_vmm():
    reason = vmm_unavailable_reason()
    if reason is not None:
        pytest.skip(f"VMM unavailable: {reason}")


def test_supported_requires_env_switch(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert not extensible_kv_cache_enabled()
    assert not extensible_kv_cache_supported()


def test_supported_false_when_vmm_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setenv("SGLANG_DISABLE_VMM", "1")
    assert not extensible_kv_cache_supported()


def test_plan_none_when_vmm_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setenv("SGLANG_DISABLE_VMM", "1")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_final_num_blocks_is_monotonic_and_capped():
    plan = ExtensibleKVCachePlan.__new__(ExtensibleKVCachePlan)
    plan.bytes_per_block = 1 << 20
    plan.max_num_blocks = 64
    low = plan.final_num_blocks(4 << 30)
    high = plan.final_num_blocks(8 << 30)
    assert low <= high <= plan.max_num_blocks
    assert plan.final_num_blocks(1 << 40) == plan.max_num_blocks


@requires_cuda
def test_plan_commit_and_free_roundtrip(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.delenv("SGLANG_DISABLE_VMM", raising=False)
    _require_vmm()
    device_index = torch.cuda.current_device()
    config = ExtensibleKVCacheConfig(
        bytes_per_block=2 << 20,
        max_num_blocks=4,
        device_index=device_index,
    )
    plan = plan_extensible_kv_cache(config)
    assert plan is not None
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
        committed = commit_extensible_kv_cache(plan, free_bytes=free_bytes)
        assert 0 <= committed <= config.max_num_blocks
        assert plan.buffer.committed_bytes <= plan.buffer.reserved_size
        # Idempotent on repeat.
        assert commit_extensible_kv_cache(plan, free_bytes=free_bytes) == committed
    finally:
        plan.free()
    assert plan.buffer.base_ptr == 0
