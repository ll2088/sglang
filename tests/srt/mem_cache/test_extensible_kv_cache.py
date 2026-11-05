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
"""Two-phase KV cache sizing: reserve, measure, commit, fall back."""

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


class _FakeBuffer:
    """Stand-in for VmmBuffer so sizing math is testable without a GPU."""

    def __init__(self, reserved_size: int) -> None:
        self.reserved_size = reserved_size
        self.committed_bytes = 0
        self.freed = False

    def commit(self, num_bytes: int) -> int:
        if self.freed:
            raise RuntimeError("VmmBuffer has been freed")
        if num_bytes < 0:
            raise ValueError("num_bytes must be non-negative")
        if num_bytes > self.reserved_size:
            raise ValueError("commit exceeds reservation")
        self.committed_bytes = max(self.committed_bytes, num_bytes)
        return self.committed_bytes

    def free(self) -> None:
        self.freed = True


def _plan(bytes_per_block: int, max_num_blocks: int, transient: int = 0):
    return ExtensibleKVCachePlan(
        buffer=_FakeBuffer(bytes_per_block * max_num_blocks),
        bytes_per_block=bytes_per_block,
        max_num_blocks=max_num_blocks,
        transient_peak_bytes=transient,
    )


def test_enabled_flag_reads_environment(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True


def test_plan_disabled_returns_none(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_final_num_blocks_respects_reservation_and_margin():
    plan = _plan(bytes_per_block=1 << 20, max_num_blocks=8)
    # Free memory far above the reservation is capped at max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 8
    # Free memory below the floor margin yields zero blocks, never negative.
    assert plan.final_num_blocks(0) == 0


def test_final_num_blocks_uses_measured_transient_peak():
    plan = _plan(bytes_per_block=1 << 20, max_num_blocks=64, transient=4 << 30)
    free = 8 << 30
    expected_margin = max(
        ekc.KV_CACHE_MARGIN_FLOOR_BYTES,
        int((4 << 30) * ekc.KV_CACHE_MARGIN_FRACTION),
    )
    expected = min((free - expected_margin) // (1 << 20), 64)
    assert plan.final_num_blocks(free) == expected


def test_commit_is_idempotent_and_monotonic():
    plan = _plan(bytes_per_block=1 << 20, max_num_blocks=8)
    assert plan.commit(3) == 3
    assert plan.commit(3) == 3
    assert plan.commit(1) == 3
    assert plan.commit(8) == 8


def test_commit_rejects_negative_blocks():
    plan = _plan(bytes_per_block=1 << 20, max_num_blocks=8)
    with pytest.raises(ValueError):
        plan.commit(-1)


def test_commit_extensible_kv_cache_uses_supplied_free_bytes():
    plan = _plan(bytes_per_block=1 << 20, max_num_blocks=8)
    committed = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
    assert committed == 8


def test_plan_falls_back_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: "no driver")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_falls_back_when_reservation_raises(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: None)

    def _boom(*args, **kwargs):
        raise VmmUnavailableError("reservation failed")

    monkeypatch.setattr(ekc, "VmmBuffer", _boom)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=8, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


@requires_cuda
def test_plan_and_commit_roundtrip_on_gpu():
    _require_vmm()
    plan = plan_extensible_kv_cache(
        ExtensibleKVCacheConfig(
            bytes_per_block=2 << 20, max_num_blocks=4, device_index=0
        )
    )
    if plan is None:
        pytest.skip("extensible KV cache disabled in this environment")
    try:
        committed = commit_extensible_kv_cache(plan)
        assert 0 <= committed <= plan.max_num_blocks
        assert plan.buffer.committed_bytes <= plan.buffer.reserved_size
    finally:
        plan.free()
