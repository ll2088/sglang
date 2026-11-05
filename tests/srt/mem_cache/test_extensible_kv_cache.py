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

"""Unit tests for the two-phase KV cache sizing helpers.

These tests do not require a GPU: they exercise the pure sizing and fallback
logic, plus the consistency helper with a fake group. The VMM-backed commit
path is covered by ``test_vmm_allocator.py`` on hosts that expose the driver
API.
"""

import pytest

from sglang.srt.mem_cache.extensible_kv_cache import (
    KV_CACHE_MARGIN_FLOOR_BYTES,
    ExtensibleKVCachePlan,
    all_ranks_support_extensible_kv_cache,
    extensible_kv_cache_enabled,
    fallback_num_blocks,
)


class _FakeBuffer:
    """Stand-in for VmmBuffer that records commits without touching a driver."""

    def __init__(self, reserved_size: int, granularity: int = 1) -> None:
        self.reserved_size = reserved_size
        self.granularity = granularity
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


class _FakeGroup:
    """Minimal all_reduce(MIN) stand-in for the consistency helper."""

    def __init__(self, values):
        self.values = list(values)

    def all_reduce(self, tensor, op=None):
        tensor.fill_(min(self.values))


def _make_plan(
    bytes_per_block: int = 1024,
    max_num_blocks: int = 100,
    transient_peak_bytes: int = 0,
) -> ExtensibleKVCachePlan:
    buffer = _FakeBuffer(bytes_per_block * max_num_blocks)
    return ExtensibleKVCachePlan(
        buffer=buffer,
        bytes_per_block=bytes_per_block,
        max_num_blocks=max_num_blocks,
        transient_peak_bytes=transient_peak_bytes,
    )


def test_enabled_defaults_to_false(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False


def test_enabled_by_env(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True


def test_final_num_blocks_respects_margin_and_cap():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    # Free memory below the floor leaves no usable blocks.
    assert plan.final_num_blocks(KV_CACHE_MARGIN_FLOOR_BYTES // 2) == 0
    # Plenty of free memory is capped by max_num_blocks.
    assert plan.final_num_blocks(1 << 40) == 100


def test_final_num_blocks_uses_measured_transient_peak():
    plan = _make_plan(
        bytes_per_block=1024,
        max_num_blocks=10_000,
        transient_peak_bytes=4 * KV_CACHE_MARGIN_FLOOR_BYTES,
    )
    free = 4 * KV_CACHE_MARGIN_FLOOR_BYTES + 1024 * 50
    # The margin is 75% of the measured peak, so only the remainder is usable.
    expected = (free - int(4 * KV_CACHE_MARGIN_FLOOR_BYTES * 0.75)) // 1024
    assert plan.final_num_blocks(free) == expected


def test_final_num_blocks_rejects_negative_free():
    plan = _make_plan()
    with pytest.raises(ValueError):
        plan.final_num_blocks(-1)


def test_commit_is_idempotent_and_monotonic():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=100)
    assert plan.commit(10) == 10
    assert plan.commit(10) == 10
    assert plan.commit(5) == 10
    assert plan.commit(20) == 20
    assert plan.committed_num_blocks() == 20


def test_commit_caps_at_max_num_blocks():
    plan = _make_plan(bytes_per_block=1024, max_num_blocks=8)
    assert plan.commit(1000) == 8


def test_commit_rejects_negative_blocks():
    plan = _make_plan()
    with pytest.raises(ValueError):
        plan.commit(-1)


def test_commit_after_free_raises():
    plan = _make_plan()
    plan.free()
    with pytest.raises(RuntimeError):
        plan.commit(1)


def test_fallback_num_blocks_matches_estimate_path():
    assert fallback_num_blocks(1024, 1024 * 50, 100) == 50
    assert fallback_num_blocks(1024, 1024 * 500, 100) == 100
    assert fallback_num_blocks(1024, 0, 100) == 0


def test_fallback_num_blocks_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        fallback_num_blocks(0, 1024, 10)
    with pytest.raises(ValueError):
        fallback_num_blocks(1024, -1, 10)
    with pytest.raises(ValueError):
        fallback_num_blocks(1024, 1024, -1)


def test_all_ranks_support_requires_unanimity():
    assert all_ranks_support_extensible_kv_cache(True, None) is True
    assert all_ranks_support_extensible_kv_cache(False, None) is False
    assert all_ranks_support_extensible_kv_cache(True, _FakeGroup([1, 1])) is True
    assert all_ranks_support_extensible_kv_cache(True, _FakeGroup([1, 0])) is False
