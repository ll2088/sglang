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
    """Stand-in for VmmBuffer so sizing logic can be tested without a GPU."""

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
            raise ValueError(
                f"requested {num_bytes} bytes exceeds reserved {self.reserved_size}"
            )
        self.committed_bytes = max(self.committed_bytes, num_bytes)
        return self.committed_bytes

    def free(self) -> None:
        self.freed = True
        self.committed_bytes = 0


def _make_plan(
    bytes_per_block: int = 1 << 20,
    max_num_blocks: int = 16,
    transient_peak_bytes: int = 0,
) -> ExtensibleKVCachePlan:
    return ExtensibleKVCachePlan(
        buffer=_FakeBuffer(bytes_per_block * max_num_blocks),
        bytes_per_block=bytes_per_block,
        max_num_blocks=max_num_blocks,
        transient_peak_bytes=transient_peak_bytes,
    )


def test_enabled_flag_reads_env(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "0")
    assert extensible_kv_cache_enabled() is False


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "0")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_returns_none_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: "no driver")
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_plan_returns_none_on_reservation_failure(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    monkeypatch.setattr(ekc, "vmm_unavailable_reason", lambda: None)

    def _boom(*args, **kwargs):
        raise VmmUnavailableError("reserve failed")

    monkeypatch.setattr(ekc, "VmmBuffer", _boom)
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(config) is None


def test_final_num_blocks_respects_cap_and_margin():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=16)
    # Free memory far above the reservation: capped at max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 16
    # Free memory below the floor margin: no blocks.
    assert plan.final_num_blocks(0) == 0
    assert plan.final_num_blocks(ekc.KV_CACHE_MARGIN_FLOOR_BYTES // 2) == 0


def test_final_num_blocks_uses_measured_transient_peak():
    plan = _make_plan(
        bytes_per_block=1 << 20,
        max_num_blocks=64,
        transient_peak_bytes=8 << 20,
    )
    free = 40 << 20
    margin = max(
        ekc.KV_CACHE_MARGIN_FLOOR_BYTES,
        int((8 << 20) * ekc.KV_CACHE_MARGIN_FRACTION),
    )
    expected = min((free - margin) // (1 << 20), 64)
    assert plan.final_num_blocks(free) == expected


def test_final_num_blocks_rejects_negative_free():
    plan = _make_plan()
    with pytest.raises(ValueError):
        plan.final_num_blocks(-1)


def test_commit_is_idempotent_and_monotonic():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=8)
    assert plan.commit(3) == 3
    assert plan.commit(3) == 3
    assert plan.commit(1) == 3
    assert plan.commit(8) == 8
    assert plan.commit(8) == 8


def test_commit_clamps_to_max_blocks():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=4)
    assert plan.commit(100) == 4


def test_commit_rejects_negative_blocks():
    plan = _make_plan()
    with pytest.raises(ValueError):
        plan.commit(-1)


def test_commit_after_free_raises():
    plan = _make_plan()
    plan.free()
    with pytest.raises(RuntimeError):
        plan.commit(1)


def test_commit_extensible_kv_cache_uses_supplied_free_bytes():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=16)
    committed = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
    assert committed == 16


def test_commit_extensible_kv_cache_is_repeatable():
    plan = _make_plan(bytes_per_block=1 << 20, max_num_blocks=16)
    first = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
    second = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
    assert first == second == 16


@requires_cuda
def test_plan_and_commit_with_real_vmm():
    _require_vmm()
    device_index = torch.cuda.current_device()
    bytes_per_block = 2 << 20
    config = ExtensibleKVCacheConfig(
        bytes_per_block=bytes_per_block,
        max_num_blocks=4,
        device_index=device_index,
    )
    plan = plan_extensible_kv_cache(config)
    if plan is None:
        pytest.skip("extensible KV cache disabled by environment")
    try:
        assert plan.reserved_bytes >= bytes_per_block * 4
        committed = commit_extensible_kv_cache(plan, free_bytes=1 << 40)
        assert committed == 4
        assert plan.buffer.committed_bytes <= plan.buffer.reserved_size
        # Idempotent second call.
        assert commit_extensible_kv_cache(plan, free_bytes=1 << 40) == 4
    finally:
        plan.free()


@requires_cuda
def test_real_vmm_commit_never_exceeds_device_memory():
    _require_vmm()
    device_index = torch.cuda.current_device()
    free_bytes, _ = torch.cuda.mem_get_info(device_index)
    # Reserve a small buffer and commit only what the device can back.
    buf = VmmBuffer(4 << 20, device_index)
    try:
        target = min(4 << 20, free_bytes // 2)
        committed = buf.commit(target)
        assert committed <= buf.reserved_size
        assert committed <= free_bytes
    finally:
        buf.free()
