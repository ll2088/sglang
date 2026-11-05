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
"""Unit tests for the minimal VMM-backed growable device buffer.

These tests exercise the bookkeeping that does not require a GPU driver by
monkeypatching the driver calls, plus the availability probe. The real driver
path is covered by the integration test in test_extensible_kv_cache.py.
"""

from __future__ import annotations

import ctypes

import pytest

from sglang.srt.mem_cache import vmm_allocator
from sglang.srt.mem_cache.vmm_allocator import (
    VmmBuffer,
    VmmUnavailableError,
    is_vmm_available,
    vmm_unavailable_reason,
)


class _FakeDriver:
    """Records driver calls and hands out monotonically increasing handles."""

    def __init__(self, granularity: int = 2 << 20, fail_on_map: int | None = None):
        self.granularity = granularity
        self.fail_on_map = fail_on_map
        self.next_handle = 1
        self.reserved: list[tuple[int, int]] = []
        self.mapped: list[tuple[int, int, int]] = []
        self.unmapped: list[tuple[int, int]] = []
        self.released: list[int] = []
        self.freed: list[tuple[int, int]] = []
        self.map_calls = 0

    def cuMemGetAllocationGranularity(self, granule, prop, flag):
        granule._obj.value = self.granularity
        return 0

    def cuMemAddressReserve(self, ptr, size, alignment, addr, flags):
        base = 0x1000_0000 + len(self.reserved) * 0x1000_0000
        ptr._obj.value = base
        self.reserved.append((base, size))
        return 0

    def cuMemCreate(self, handle, size, prop, flags):
        handle._obj.value = self.next_handle
        self.next_handle += 1
        return 0

    def cuMemMap(self, va, size, handle, offset):
        self.map_calls += 1
        if self.fail_on_map is not None and self.map_calls == self.fail_on_map:
            return 1
        self.mapped.append((va, size, handle))
        return 0

    def cuMemSetAccess(self, va, size, desc, count):
        return 0

    def cuMemUnmap(self, va, size):
        self.unmapped.append((va, size))
        return 0

    def cuMemRelease(self, handle):
        self.released.append(handle)
        return 0

    def cuMemAddressFree(self, ptr, size):
        self.freed.append((ptr, size))
        return 0


@pytest.fixture
def fake_driver(monkeypatch):
    driver = _FakeDriver()
    monkeypatch.setattr(vmm_allocator, "_load_driver", lambda: driver)
    monkeypatch.setattr(vmm_allocator, "vmm_unavailable_reason", lambda: None)
    monkeypatch.setattr(vmm_allocator.torch.cuda, "current_device", lambda: 0)
    return driver


def test_unavailable_reason_respects_disable_env(monkeypatch):
    monkeypatch.setenv("SGLANG_DISABLE_VMM", "1")
    assert vmm_unavailable_reason() == "SGLANG_DISABLE_VMM=1"
    assert not is_vmm_available()


def test_unavailable_reason_without_cuda(monkeypatch):
    monkeypatch.delenv("SGLANG_DISABLE_VMM", raising=False)
    monkeypatch.setattr(vmm_allocator.torch.cuda, "is_available", lambda: False)
    assert vmm_unavailable_reason() == "CUDA is not available"


def test_reserve_rounds_up_to_granularity(fake_driver):
    buf = VmmBuffer(3 << 20, 0)
    try:
        assert buf.granularity == 2 << 20
        assert buf.reserved_size == 4 << 20
        assert buf.committed_bytes == 0
        assert fake_driver.reserved == [(buf.base_ptr, 4 << 20)]
    finally:
        buf.free()


def test_commit_is_monotonic_and_idempotent(fake_driver):
    buf = VmmBuffer(8 << 20, 0)
    try:
        assert buf.commit(1) == 2 << 20
        assert buf.commit(1) == 2 << 20
        assert buf.commit(3 << 20) == 4 << 20
        assert buf.commit(2 << 20) == 4 << 20
        assert buf.num_committed_granules == 2
    finally:
        buf.free()


def test_commit_never_exceeds_reservation(fake_driver):
    buf = VmmBuffer(4 << 20, 0)
    try:
        with pytest.raises(ValueError):
            buf.commit((4 << 20) + 1)
        assert buf.committed_bytes == 0
    finally:
        buf.free()


def test_commit_range_maps_only_missing_granules(fake_driver):
    buf = VmmBuffer(8 << 20, 0)
    try:
        buf.commit_range(0, 2 << 20)
        assert buf.num_committed_granules == 1
        buf.commit_range(0, 4 << 20)
        assert buf.num_committed_granules == 2
        assert fake_driver.map_calls == 2
    finally:
        buf.free()


def test_failed_commit_rolls_back(monkeypatch):
    driver = _FakeDriver(fail_on_map=2)
    monkeypatch.setattr(vmm_allocator, "_load_driver", lambda: driver)
    monkeypatch.setattr(vmm_allocator, "vmm_unavailable_reason", lambda: None)
    monkeypatch.setattr(vmm_allocator.torch.cuda, "current_device", lambda: 0)
    buf = VmmBuffer(8 << 20, 0)
    try:
        with pytest.raises(VmmUnavailableError):
            buf.commit(4 << 20)
        # The first granule was mapped, then rolled back on failure.
        assert buf.committed_bytes == 0
        assert len(driver.unmapped) == 1
        assert len(driver.released) == 2
    finally:
        buf.free()


def test_free_releases_reservation(fake_driver):
    buf = VmmBuffer(4 << 20, 0)
    buf.commit(2 << 20)
    base = buf.base_ptr
    buf.free()
    assert buf.is_freed
    assert fake_driver.freed == [(base, 4 << 20)]
    # Freeing twice is a no-op.
    buf.free()
    assert len(fake_driver.freed) == 1


def test_commit_after_free_raises(fake_driver):
    buf = VmmBuffer(4 << 20, 0)
    buf.free()
    with pytest.raises(RuntimeError):
        buf.commit(1)


def test_invalid_max_bytes_rejected(fake_driver):
    with pytest.raises(ValueError):
        VmmBuffer(0, 0)
