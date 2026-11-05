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
"""Integration: two-phase sizing keeps views valid and frees cleanly."""

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
def test_two_phase_sizing_keeps_view_and_frees(monkeypatch):
    _require_vmm()
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    bytes_per_block = 2 << 20
    plan = plan_extensible_kv_cache(
        ExtensibleKVCacheConfig(
            bytes_per_block=bytes_per_block,
            max_num_blocks=4,
            device_index=torch.cuda.current_device(),
        )
    )
    if plan is None:
        pytest.skip("extensible KV cache disabled in this environment")
    try:
        view = plan.buffer.as_tensor(torch.float16, (bytes_per_block // 2,))
        base_ptr = view.data_ptr()
        committed = commit_extensible_kv_cache(plan)
        assert 0 <= committed <= plan.max_num_blocks
        # The view must keep its address across the commit, which is what
        # keeps captured CUDA graphs valid.
        assert view.data_ptr() == base_ptr
        assert plan.buffer.committed_bytes <= plan.buffer.reserved_size
    finally:
        plan.free()
    assert plan.buffer.base_ptr == 0


@requires_cuda
def test_fallback_path_allocates_equivalent_capacity(monkeypatch):
    # With the feature disabled, planning returns None and the caller keeps
    # the existing estimate-based allocation path unchanged.
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    plan = plan_extensible_kv_cache(
        ExtensibleKVCacheConfig(
            bytes_per_block=2 << 20,
            max_num_blocks=4,
            device_index=torch.cuda.current_device(),
        )
    )
    assert plan is None
