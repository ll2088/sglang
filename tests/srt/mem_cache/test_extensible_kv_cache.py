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

import os
from unittest import mock

import pytest
import torch

from sglang.srt.mem_cache import extensible_kv_cache as ekc
from sglang.srt.mem_cache.extensible_kv_cache import (
    ExtensibleKVCacheConfig,
    ExtensibleKVCachePlan,
    all_ranks_support_extensible_kv_cache,
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


def test_enabled_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True


def test_plan_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    assert plan_extensible_kv_cache(cfg) is None


def test_plan_returns_none_when_vmm_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    cfg = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20, max_num_blocks=4, device_index=0
    )
    with mock.patch.object(ekc, "vmm_unavailable_reason", return_value="nope"):
        assert plan_extensible_kv_cache(cfg) is None


def test_final_num_blocks_respects_margin_and_cap():
    plan = ExtensibleKVCachePlan(
        buffer=mock.Mock(reserved_size=1 << 30),
        bytes_per_block=1 << 20,
        max_num_blocks=100,
        transient_peak_bytes=0,
    )
    # Free memory below the floor yields zero blocks, never negative.
    assert plan.final_num_blocks(0) == 0
    assert plan.final_num_blocks(ekc.KV_CACHE_MARGIN_FLOOR_BYTES // 2) == 0
    # Plenty of free memory is capped by max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 100


def test_final_num_blocks_scales_with_transient_peak():
    plan = ExtensibleKVCachePlan(
        buffer=mock.Mock(reserved_size=1 << 30),
        bytes_per_block=1 << 20,
        max_num_blocks=1000,
        transient_peak_bytes=4 << 30,
    )
    free = 8 << 30
    expected_margin = int((4 << 30) * ekc.KV_CACHE_MARGIN_FRACTION)
    expected = (free - expected_margin) // (1 << 20)
    assert plan.final_num_blocks(free) == expected


def test_commit_is_idempotent_and_monotonic():
    buffer = mock.Mock()
    buffer.committed_bytes = 0

    def _commit(nbytes):
        buffer.committed_bytes = max(buffer.committed_bytes, nbytes)
        return buffer.committed_bytes

    buffer.commit.side_effect = _commit
    plan = ExtensibleKVCachePlan(
        buffer=buffer,
        bytes_per_block=1 << 20,
        max_num_blocks=8,
    )
    assert plan.commit(3) == 3
    assert plan.commit(3) == 3
    assert plan.commit(1) == 3
    assert plan.commit(8) == 8
    assert plan.committed_num_blocks() == 8


def test_commit_rejects_negative():
    plan = ExtensibleKVCachePlan(
        buffer=mock.Mock(committed_bytes=0),
        bytes_per_block=1 << 20,
        max_num_blocks=4,
    )
    with pytest.raises(ValueError):
        plan.commit(-1)


def test_commit_extensible_uses_measured_free_bytes():
    buffer = mock.Mock()
    buffer.committed_bytes = 0
    buffer.commit.side_effect = lambda n: setattr(
        buffer, "committed_bytes", n
    ) or n
    plan = ExtensibleKVCachePlan(
        buffer=buffer,
        bytes_per_block=1 << 20,
        max_num_blocks=100,
        transient_peak_bytes=0,
    )
    free = ekc.KV_CACHE_MARGIN_FLOOR_BYTES + (10 << 20)
    assert commit_extensible_kv_cache(plan, free_bytes=free) == 10
    # Repeating with the same measurement is a no-op.
    assert commit_extensible_kv_cache(plan, free_bytes=free) == 10


def test_commit_extensible_propagates_mapping_failure():
    buffer = mock.Mock()
    buffer.commit.side_effect = VmmUnavailableError("boom")
    plan = ExtensibleKVCachePlan(
        buffer=buffer,
        bytes_per_block=1 << 20,
        max_num_blocks=4,
    )
    with pytest.raises(VmmUnavailableError):
        commit_extensible_kv_cache(plan, free_bytes=1 << 30)


def test_all_ranks_support_without_group_returns_local():
    assert all_ranks_support_extensible_kv_cache(True) is True
    assert all_ranks_support_extensible_kv_cache(False) is False


def test_all_ranks_support_uses_min_reduce():
    class _Group:
        def __init__(self, value):
            self.value = value
            self.calls = []

        def all_reduce(self, tensor, op=None):
            self.calls.append(op)
            tensor.fill_(self.value)

    group = _Group(0)
    assert all_ranks_support_extensible_kv_cache(True, group) is False
    assert group.calls == [torch.distributed.ReduceOp.MIN]

    group = _Group(1)
    assert all_ranks_support_extensible_kv_cache(True, group) is True


@requires_cuda
def test_plan_and_commit_roundtrip():
    _require_vmm()
    os.environ["SGLANG_ENABLE_EXTENSIBLE_KV_CACHE"] = "1"
    try:
        cfg = ExtensibleKVCacheConfig(
            bytes_per_block=2 << 20,
            max_num_blocks=4,
            device_index=torch.cuda.current_device(),
        )
        plan = plan_extensible_kv_cache(cfg)
        assert plan is not None
        try:
            free, _ = torch.cuda.mem_get_info(cfg.device_index)
            committed = commit_extensible_kv_cache(plan, free_bytes=free)
            assert 0 <= committed <= cfg.max_num_blocks
            assert plan.committed_num_blocks() == committed
            # A second commit with the same measurement is idempotent.
            assert commit_extensible_kv_cache(plan, free_bytes=free) == committed
        finally:
            plan.free()
    finally:
        os.environ.pop("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", None)


@requires_cuda
def test_vmm_buffer_rolls_back_failed_commit():
    _require_vmm()
    buf = VmmBuffer(8 << 20, torch.cuda.current_device())
    try:
        buf.commit(2 << 20)
        before = buf.committed_bytes
        with mock.patch.object(
            buf, "_map_granule", side_effect=VmmUnavailableError("boom")
        ):
            with pytest.raises(VmmUnavailableError):
                buf.commit(6 << 20)
        assert buf.committed_bytes == before
    finally:
        buf.free()
