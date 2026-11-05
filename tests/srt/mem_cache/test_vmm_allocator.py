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
"""CPU-safe unit tests for the VMM allocator wrapper.

These tests do not require a GPU: they exercise the availability probe, the
fallback reason, and the pure-Python sizing helpers. GPU-backed reservation
and commit behaviour is covered by the extensible KV cache tests, which skip
when VMM is unavailable.
"""

import os
from unittest import mock

import pytest

from sglang.srt.mem_cache import vmm_allocator
from sglang.srt.mem_cache.vmm_allocator import (
    VmmBuffer,
    VmmUnavailableError,
    is_vmm_available,
    vmm_unavailable_reason,
)


def test_disable_env_reports_reason(monkeypatch):
    monkeypatch.setenv("SGLANG_DISABLE_VMM", "1")
    assert vmm_unavailable_reason() == "SGLANG_DISABLE_VMM=1"
    assert is_vmm_available() is False


def test_unavailable_without_cuda(monkeypatch):
    monkeypatch.delenv("SGLANG_DISABLE_VMM", raising=False)
    with mock.patch.object(vmm_allocator.torch.cuda, "is_available", return_value=False):
        assert vmm_unavailable_reason() == "CUDA is not available"
        assert is_vmm_available() is False


def test_buffer_rejects_when_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_DISABLE_VMM", "1")
    with pytest.raises(VmmUnavailableError):
        VmmBuffer(1 << 20, 0)


def test_buffer_rejects_non_positive_size(monkeypatch):
    monkeypatch.delenv("SGLANG_DISABLE_VMM", raising=False)
    with mock.patch.object(vmm_allocator, "vmm_unavailable_reason", return_value=None):
        with pytest.raises(ValueError):
            VmmBuffer(0, 0)


def test_round_up_is_granule_aligned():
    granule = 2 << 20
    for value in (0, 1, granule - 1, granule, granule + 1, 3 * granule + 7):
        rounded = vmm_allocator._round_up(value, granule)
        assert rounded % granule == 0
        assert rounded >= value
        assert rounded - value < granule


def test_commit_is_monotonic_and_bounded():
    """Exercise the commit bookkeeping without a driver by stubbing mapping."""
    granule = 2 << 20
    buffer = VmmBuffer.__new__(VmmBuffer)
    buffer._freed = False
    buffer.device_index = 0
    buffer.granularity = granule
    buffer.reserved_size = 4 * granule
    buffer.base_ptr = 0x1000
    buffer._mapped_granules = set()
    buffer._handles = []

    def fake_map(granule_index):
        buffer._mapped_granules.add(granule_index)
        buffer._handles.append((granule_index, buffer.base_ptr + granule_index * granule, granule))

    def fake_unmap(granule_index):
        buffer._mapped_granules.discard(granule_index)
        buffer._handles = [h for h in buffer._handles if h[0] != granule_index]

    buffer._map_granule = fake_map
    buffer._unmap_granule = fake_unmap

    assert buffer.commit(0) == 0
    assert buffer.commit(granule) == granule
    # Committing fewer bytes is a no-op.
    assert buffer.commit(granule // 2) == granule
    assert buffer.commit(3 * granule) == 3 * granule
    assert buffer.commit(4 * granule) == 4 * granule
    with pytest.raises(ValueError):
        buffer.commit(4 * granule + 1)


def test_commit_rolls_back_on_failure():
    granule = 2 << 20
    buffer = VmmBuffer.__new__(VmmBuffer)
    buffer._freed = False
    buffer.device_index = 0
    buffer.granularity = granule
    buffer.reserved_size = 4 * granule
    buffer.base_ptr = 0x1000
    buffer._mapped_granules = set()
    buffer._handles = []

    calls = {"n": 0}

    def flaky_map(granule_index):
        calls["n"] += 1
        if calls["n"] == 2:
            raise VmmUnavailableError("boom")
        buffer._mapped_granules.add(granule_index)
        buffer._handles.append((granule_index, buffer.base_ptr + granule_index * granule, granule))

    def fake_unmap(granule_index):
        buffer._mapped_granules.discard(granule_index)
        buffer._handles = [h for h in buffer._handles if h[0] != granule_index]

    buffer._map_granule = flaky_map
    buffer._unmap_granule = fake_unmap

    with pytest.raises(VmmUnavailableError):
        buffer.commit(3 * granule)
    assert buffer.committed_bytes == 0
    assert buffer._handles == []
