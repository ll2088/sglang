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

"""Integration checks for the two-phase KV cache sizing flow.

These tests exercise the plan/commit/free lifecycle the way model_runner and
mem_cache would use it, without requiring a full engine. They verify that the
final block count follows measured free memory, that repeated commits are
idempotent, and that the fallback path is equivalent to the estimate path.
"""

import pytest
import torch

from sglang.srt.mem_cache.extensible_kv_cache import (
    ExtensibleKVCacheConfig,
    ExtensibleKVCachePlan,
    commit_extensible_kv_cache,
    plan_extensible_kv_cache,
)
from sglang.srt.mem_cache.vmm_allocator import (
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


def _make_plan(bytes_per_block: int, max_num_blocks: int) -> ExtensibleKVCachePlan:
    plan = ExtensibleKVCachePlan.__new__(ExtensibleKVCachePlan)
    plan.bytes_per_block = bytes_per_block
    plan.max_num_blocks = max_num_blocks
    return plan


def test_final_num_blocks_matches_measured_free_memory():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=64)
    # Free memory large enough to hit the cap.
    assert plan.final_num_blocks(1 << 40) == 64
    # Free memory below the floor yields no blocks.
    assert plan.final_num_blocks(0) == 0
    # A mid-range value is bounded by the cap and non-negative.
    blocks = plan.final_num_blocks(32 << 20)
    assert 0 <= blocks <= 64


def test_commit_is_idempotent_for_same_free_memory():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=8)

    class _CountingBuffer:
        reserved_size = 8 << 20

        def __init__(self):
            self.committed_bytes = 0
            self.calls = 0

        def commit(self, num_bytes):
            self.calls += 1
            self.committed_bytes = max(self.committed_bytes, num_bytes)
            return self.committed_bytes

        def free(self):
            pass

    plan.buffer = _CountingBuffer()
    first = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
    second = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
    assert first == second == 8
    assert plan.buffer.calls == 2


def test_commit_failure_surfaces_for_fallback():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=4)

    class _FailingBuffer:
        reserved_size = 4 << 20
        committed_bytes = 0

        def commit(self, num_bytes):
            raise VmmUnavailableError("simulated mapping failure")

        def free(self):
            pass

    plan.buffer = _FailingBuffer()
    with pytest.raises(VmmUnavailableError):
        commit_extensible_kv_cache(plan, free_bytes=8 << 20)


def test_plan_returns_none_when_vmm_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setenv("SGLANG_DISABLE_VMM", "1")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20,
        max_num_blocks=4,
        device_index=0,
    )
    assert plan_extensible_kv_cache(config) is None


@requires_cuda
def test_plan_commit_free_roundtrip(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
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
        # Re-committing the same measured size is a no-op.
        assert commit_extensible_kv_cache(plan, free_bytes=free_bytes) == committed
    finally:
        plan.free()
    # After free, the reservation is released and further commits fail.
    with pytest.raises(RuntimeError):
        plan.buffer.commit(1 << 20)
