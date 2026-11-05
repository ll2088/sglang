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
"""Integration tests for the two-phase KV cache sizing flow.

These tests exercise the plan/commit/free lifecycle without a model runner so
they run on CPU-only CI. The CUDA-specific paths are covered by
``test_vmm_allocator.py`` and ``test_extensible_kv_cache.py``.
"""

import pytest

from sglang.srt.mem_cache.extensible_kv_cache import (
    ExtensibleKVCacheConfig,
    ExtensibleKVCachePlan,
    commit_extensible_kv_cache,
    plan_extensible_kv_cache,
)
from sglang.srt.mem_cache.vmm_allocator import VmmUnavailableError


class _FakeBuffer:
    """In-memory stand-in for VmmBuffer used to test the sizing math."""

    def __init__(self, reserved_size: int) -> None:
        self.reserved_size = reserved_size
        self.committed_bytes = 0
        self.freed = False

    def commit(self, num_bytes: int) -> int:
        if self.freed:
            raise RuntimeError("freed")
        if num_bytes > self.reserved_size:
            raise ValueError("exceeds reservation")
        self.committed_bytes = max(self.committed_bytes, num_bytes)
        return self.committed_bytes

    def free(self) -> None:
        self.freed = True
        self.committed_bytes = 0


def _make_plan(bytes_per_block: int, max_num_blocks: int) -> ExtensibleKVCachePlan:
    plan = ExtensibleKVCachePlan.__new__(ExtensibleKVCachePlan)
    plan.bytes_per_block = bytes_per_block
    plan.max_num_blocks = max_num_blocks
    plan.buffer = _FakeBuffer(bytes_per_block * max_num_blocks)
    return plan


def test_commit_matches_measured_free_memory():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=100)
    # 64 MiB free, margin floor is 1 GiB -> no usable blocks.
    assert commit_extensible_kv_cache(plan, free_bytes=64 << 20) == 0
    # 8 GiB free: margin = max(1 GiB, 0.75 * 8 GiB) = 6 GiB, usable 2 GiB.
    assert commit_extensible_kv_cache(plan, free_bytes=8 << 30) == 2048


def test_commit_is_idempotent():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=100)
    first = commit_extensible_kv_cache(plan, free_bytes=8 << 30)
    second = commit_extensible_kv_cache(plan, free_bytes=8 << 30)
    assert first == second


def test_commit_never_exceeds_cap():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=10)
    assert commit_extensible_kv_cache(plan, free_bytes=1 << 40) == 10


def test_free_releases_reservation():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=10)
    commit_extensible_kv_cache(plan, free_bytes=8 << 30)
    plan.free()
    assert plan.buffer.freed is True
    assert plan.buffer.committed_bytes == 0


def test_commit_failure_surfaces_for_fallback():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=10)

    class _FailingBuffer(_FakeBuffer):
        def commit(self, num_bytes: int) -> int:
            raise VmmUnavailableError("simulated mapping failure")

    plan.buffer = _FailingBuffer(10 << 20)
    with pytest.raises(VmmUnavailableError):
        commit_extensible_kv_cache(plan, free_bytes=8 << 30)


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_returns_none_when_vmm_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setenv("SGLANG_DISABLE_VMM", "1")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None
