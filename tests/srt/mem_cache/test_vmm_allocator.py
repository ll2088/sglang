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

These tests exercise the pure-Python bookkeeping of ``VmmBuffer`` without a
GPU by monkeypatching the driver calls. The real driver path is covered by the
integration tests that skip when VMM is unavailable.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache import vmm_allocator


class _FakeDriver:
    """Records driver calls and hands out deterministic handles."""

    def __init__(self, granularity: int = 2 << 20) -> None:
        self.granularity = granularity
        self.reserved: list[tuple[int, int]] = []
        self.mapped: list[tuple[int, int, int]] = []
        self.released: list[int] = []
        self.unmapped: list[tuple[int, int]] = []
        self.fail_on_map: int | None = None
        self._next_handle = 1
        self._next_ptr = 0x1000_0000

    def cuMemGetAllocationGranularity(self, granule, prop, flag):
        granule._obj.value = self.granularity
        return 0

    def cuMemAddressReserve(self, ptr, size, alignment, addr, flags):
        base = self._next_ptr
        self._next_ptr += size + self.granularity
        ptr._obj.value = base
        self.reserved.append((base, size))
        return 0

    def cuMemCreate(self, handle, size, prop, flags):
        handle._obj.value = self._next_handle
        self._next_handle += 1
        return 0

    def cuMemMap(self, va, size, handle, flags):
        if self.fail_on_map is not None and len(self.mapped) >= self.fail_on_map:
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
        return 0


@pytest.fixture
def fake_driver(monkeypatch):
    driver = _FakeDriver()
    monkeypatch.setattr(vmm_allocator, "_load_driver", lambda: driver)
    monkeypatch.setattr(vmm_allocator, "vmm_unavailable_reason", lambda: None)
    monkeypatch.setattr(vmm_allocator.torch.cuda, "current_device", lambda: 0)
    return driver


def test_commit_is_monotonic_and_bounded(fake_driver):
    buf = vmm_allocator.VmmBuffer(8 << 20, 0)
    assert buf.committed_bytes == 0

    committed = buf.commit(3 << 20)
    assert committed == 4 << 20
    assert buf.num_committed_granules == 2

    # Committing fewer bytes is a no-op.
    assert buf.commit(1 << 20) == committed
    assert buf.num_committed_granules == 2

    # Committing more maps only the missing granules.
    assert buf.commit(8 << 20) == 8 << 20
    assert buf.num_committed_granules == 4
    assert len(fake_driver.mapped) == 4


def test_commit_rejects_oversized_request(fake_driver):
    buf = vmm_allocator.VmmBuffer(4 << 20, 0)
    with pytest.raises(ValueError):
        buf.commit(8 << 20)
    assert buf.committed_bytes == 0


def test_commit_range_maps_only_requested_granules(fake_driver):
    buf = vmm_allocator.VmmBuffer(8 << 20, 0)
    buf.commit_range(2 << 20, 4 << 20)
    assert buf.num_committed_granules == 1
    assert buf.committed_bytes == 2 << 20


def test_failed_commit_rolls_back(fake_driver):
    buf = vmm_allocator.VmmBuffer(8 << 20, 0)
    fake_driver.fail_on_map = 1
    with pytest.raises(vmm_allocator.VmmUnavailableError):
        buf.commit(8 << 20)
    assert buf.committed_bytes == 0
    assert fake_driver.released


def test_free_releases_reservation(fake_driver):
    buf = vmm_allocator.VmmBuffer(4 << 20, 0)
    buf.commit(4 << 20)
    buf.free()
    assert buf.is_freed
    assert buf.committed_bytes == 0
    assert fake_driver.unmapped


def test_vmm_unavailable_reason_respects_disable_env(monkeypatch):
    monkeypatch.setenv("SGLANG_DISABLE_VMM", "1")
    assert vmm_allocator.vmm_unavailable_reason() == "SGLANG_DISABLE_VMM=1"
