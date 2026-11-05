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
"""Two-phase sizing and fallback behaviour for the extensible KV cache."""

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
    def __init__(self, size: int) -> None:
        self.reserved_size = size
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


def _plan(max_blocks=16, bytes_per_block=1 << 20, transient=0):
    return ExtensibleKVCachePlan(
        buffer=_FakeBuffer(max_blocks * bytes_per_block),
        bytes_per_block=bytes_per_block,
        max_num_blocks=max_blocks,
        transient_peak_bytes=transient,
    )


def test_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True


def test_final_num_blocks_respects_margin_and_cap():
    plan = _plan(max_blocks=16, bytes_per_block=1 << 20)
    # Free memory below the floor leaves nothing usable.
    assert plan.final_num_blocks(0) == 0
    # Plenty of free memory is capped by the reservation.
    assert plan.final_num_blocks(1 << 40) == 16


def test_final_num_blocks_uses_measured_transient():
    plan = _plan(max_blocks=64, bytes_per_block=1 << 20, transient=8 << 30)
    free = 16 << 30
    blocks = plan.final_num_blocks(free)
    margin = max(ekc.KV_CACHE_MARGIN_FLOOR_BYTES, int((8 << 30) * ekc.KV_CACHE_MARGIN_FRACTION))
    assert blocks == (free - margin) // (1 << 20)


def test_commit_is_idempotent_and_monotonic():
    plan = _plan(max_blocks=16, bytes_per_block=1 << 20)
    assert plan.commit(4) == 4
    assert plan.commit(4) == 4
    assert plan.commit(2) == 4
    assert plan.commit(100) == 16


def test_commit_rejects_negative():
    plan = _plan()
    with pytest.raises(ValueError):
        plan.commit(-1)


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(cfg) is None


def test_plan_returns_none_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: "forced")
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(cfg) is None


def test_plan_falls_back_on_reservation_error(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: None)

    def _boom(*args, **kwargs):
        raise VmmUnavailableError("boom")

    monkeypatch.setattr(ekc, "VmmBuffer", _boom)
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(cfg) is None


def test_commit_uses_supplied_free_bytes():
    plan = _plan(max_blocks=16, bytes_per_block=1 << 20)
    committed = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
    assert committed == 16


@requires_cuda
def test_vmm_plan_commit_and_free():
    _require_vmm()
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=2 << 20,
        max_num_blocks=4,
        device_index=torch.cuda.current_device(),
    )
    plan = plan_extensible_kv_cache(cfg)
    if plan is None:
        pytest.skip("reservation unavailable")
    try:
        committed = plan.commit(2)
        assert committed == 2
        assert plan.buffer.committed_bytes <= plan.buffer.reserved_size
    finally:
        plan.free()
        assert plan.buffer.base_ptr == 0
