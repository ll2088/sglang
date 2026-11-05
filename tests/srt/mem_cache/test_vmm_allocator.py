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
"""Unit tests for the minimal VMM-backed growable buffer.

These tests exercise the pure-Python bookkeeping of ``VmmBuffer`` without a
GPU: the driver calls are replaced by a fake so the reservation/commit logic,
idempotency and rollback behaviour can be checked on CPU-only CI hosts.
"""

from __future__ import annotations

import ctypes

import pytest

from sglang.srt.mem_cache import vmm_allocator


class _FakeDriver:
    """Records driver calls and simulates a fixed granularity."""

    def __init__(self, granularity: int = 2 << 20, fail_on_map: int | None = None):
        self.granularity = granularity
        self.fail_on_map = fail_on_map
        self.reserved: list[int] = []
        self.mapped: list[int] = []
        self.unmapped: list[int] = []
        self.released: list[int] = []
        self._next_handle = 1

    def cuMemGetAllocationGranularity(self, granule, prop, flag):
        granule._obj.value = self.granularity
        return 0

    def cuMemAddressReserve(self, ptr, size, alignment, addr, flags):
        ptr._obj.value = 0x1000_0000
        self.reserved.append(size)
        return 0

    def cuMemCreate(self, handle, size, prop, flags):
        handle._obj.value = self._next_handle
        self._next_handle += 1
        return 0

    def cuMemMap(self, va, size, handle, flags):
        if self.fail_on_map is not None and len(self.mapped) >= self.fail_on_map:
            return 1
        self.mapped.append(va)
        return 0

    def cuMemSetAccess(self, va, size, desc, count):
        return 0

    def cuMemUnmap(self, va, size):
        self.unmapped.append(va)
        return 0

    def cuMemRelease(self, handle):
        self.released.append(handle)
        return 0

    def cuMemAddressFree(self, ptr, size):
        return 0


@pytest.fixture
def fake_driver(monkeypatch):
    driver = _FakeDriver()
    monkeypatch.setattr(vmm_allocator, "_load_driver", lambda: driver)
    monkeypatch.setattr(vmm_allocator, "vmm_unavailable_reason", lambda: None)
    monkeypatch.setattr(vmm_allocator.torch.cuda, "current_device", lambda: 0)
    return driver


def test_reserve_rounds_up_to_granularity(fake_driver):
    buf = vmm_allocator.VmmBuffer(3 << 20, 0)
    try:
        assert buf.reserved_size == 4 << 20
        assert buf.committed_bytes == 0
        assert fake_driver.reserved == [4 << 20]
    finally:
        buf.free()


def test_commit_is_monotonic_and_idempotent(fake_driver):
    buf = vmm_allocator.VmmBuffer(8 << 20, 0)
    try:
        assert buf.commit(1) == 2 << 20
        assert buf.commit(1) == 2 << 20
        assert buf.commit(3 << 20) == 4 << 20
        # Committing fewer bytes than already committed is a no-op.
        assert buf.commit(1 << 20) == 4 << 20
        assert buf.num_committed_granules == 2
    finally:
        buf.free()


def test_commit_never_exceeds_reservation(fake_driver):
    buf = vmm_allocator.VmmBuffer(4 << 20, 0)
    try:
        with pytest.raises(ValueError):
            buf.commit((4 << 20) + 1)
        assert buf.committed_bytes == 0
    finally:
        buf.free()


def test_commit_range_maps_only_requested_granules(fake_driver):
    buf = vmm_allocator.VmmBuffer(8 << 20, 0)
    try:
        buf.commit_range(2 << 20, 4 << 20)
        assert buf.num_committed_granules == 1
        assert buf.committed_bytes == 2 << 20
    finally:
        buf.free()


def test_failed_commit_rolls_back(monkeypatch):
    driver = _FakeDriver(fail_on_map=1)
    monkeypatch.setattr(vmm_allocator, "_load_driver", lambda: driver)
    monkeypatch.setattr(vmm_allocator, "vmm_unavailable_reason", lambda: None)
    monkeypatch.setattr(vmm_allocator.torch.cuda, "current_device", lambda: 0)
    buf = vmm_allocator.VmmBuffer(8 << 20, 0)
    try:
        with pytest.raises(vmm_allocator.VmmUnavailableError):
            buf.commit(4 << 20)
        # The first granule was mapped, then rolled back.
        assert buf.committed_bytes == 0
        assert len(driver.unmapped) == 1
        assert len(driver.released) == 1
    finally:
        buf.free()


def test_free_releases_everything(fake_driver):
    buf = vmm_allocator.VmmBuffer(4 << 20, 0)
    buf.commit(4 << 20)
    buf.free()
    assert buf.is_freed
    assert buf.committed_bytes == 0
    assert len(fake_driver.unmapped) == 2
    assert len(fake_driver.released) == 2


def test_unavailable_reason_short_circuits(monkeypatch):
    monkeypatch.setenv("SGLANG_DISABLE_VMM", "1")
    assert vmm_allocator.vmm_unavailable_reason() == "SGLANG_DISABLE_VMM=1"
    assert not vmm_allocator.is_vmm_available()
