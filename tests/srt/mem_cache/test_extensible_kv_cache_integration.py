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
    commit_extensible_kv_cache,
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


@requires_cuda
def test_two_phase_sizing_allocates_and_frees_without_leak(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    _require_vmm()
    device_index = torch.cuda.current_device()
    free_before, _ = torch.cuda.mem_get_info(device_index)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=2 << 20,
        max_num_blocks=8,
        device_index=device_index,
    )
    plan = plan_extensible_kv_cache(config)
    assert plan is not None
    try:
        committed = commit_extensible_kv_cache(plan)
        assert 0 <= committed <= config.max_num_blocks
        # Committing again must be idempotent.
        assert commit_extensible_kv_cache(plan) == committed
    finally:
        plan.free()
    free_after, _ = torch.cuda.mem_get_info(device_index)
    # The reservation is released; allow a small tolerance for allocator noise.
    assert free_after + (1 << 20) >= free_before


@requires_cuda
def test_fallback_when_vmm_disabled_matches_estimate_path(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setenv("SGLANG_DISABLE_VMM", "1")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20,
        max_num_blocks=4,
        device_index=torch.cuda.current_device(),
    )
    # With VMM disabled the planner returns None and callers keep the
    # existing estimate-based allocation path unchanged.
    assert plan_extensible_kv_cache(config) is None


@requires_cuda
def test_commit_failure_falls_back(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    _require_vmm()
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20,
        max_num_blocks=4,
        device_index=torch.cuda.current_device(),
    )
    plan = plan_extensible_kv_cache(config)
    assert plan is not None
    try:
        # A negative free-memory measurement must be rejected rather than
        # silently committing an invalid block count.
        with pytest.raises(ValueError):
            plan.final_num_blocks(-1)
    finally:
        plan.free()
