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
"""Integration tests for the two-phase sizing flow.

These tests skip when the driver VMM API is unavailable so they can run on
CPU-only CI hosts without failing.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.mem_cache import extensible_kv_cache as ekc
from sglang.srt.mem_cache.vmm_allocator import vmm_unavailable_reason


@pytest.fixture
def vmm():
    if (reason := vmm_unavailable_reason()) is not None:
        pytest.skip(f"VMM unavailable: {reason}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_two_phase_sizing_commits_within_reservation(vmm, monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    bytes_per_block = 2 << 20
    max_blocks = 8
    plan = ekc.plan_extensible_kv_cache(
        ekc.ExtensibleKVCacheConfig(
            bytes_per_block=bytes_per_block,
            max_num_blocks=max_blocks,
            device_index=torch.cuda.current_device(),
        )
    )
    assert plan is not None
    try:
        free_bytes, _ = torch.cuda.mem_get_info(plan.buffer.device_index)
        committed = ekc.commit_extensible_kv_cache(plan, free_bytes=free_bytes)
        assert 0 <= committed <= max_blocks
        assert plan.buffer.committed_bytes <= plan.buffer.reserved_size
        # Idempotent: a second commit does not shrink the reservation.
        assert ekc.commit_extensible_kv_cache(plan, free_bytes=free_bytes) == committed
    finally:
        plan.free()
    assert plan.buffer.is_freed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fallback_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: "forced")
    assert ekc.plan_extensible_kv_cache(
        ekc.ExtensibleKVCacheConfig(1 << 20, 4, 0)
    ) is None
    # The estimate-based fallback still produces a bounded block count.
    assert ekc.fallback_num_blocks(1 << 20, 4 << 20, 4) == 4
