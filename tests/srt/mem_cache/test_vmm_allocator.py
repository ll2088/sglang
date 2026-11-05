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

These tests exercise the pure-Python bookkeeping (granularity rounding,
commit monotonicity, range validation and rollback) without requiring a GPU.
The driver-facing paths are covered by ``vmm_unavailable_reason`` returning a
reason on CPU-only hosts, which is the fallback contract callers rely on.
"""

import pytest

from sglang.srt.mem_cache import vmm_allocator
from sglang.srt.mem_cache.vmm_allocator import (
    VmmBuffer,
    VmmUnavailableError,
    is_vmm_available,
    vmm_unavailable_reason,
)


class _FakeBuffer(VmmBuffer):
    """VmmBuffer with the driver calls stubbed out.

    Only the bookkeeping in ``commit``/``commit_range``/``free`` is exercised;
    ``_map_granule`` records the granule index instead of calling the driver.
    """

    def __init__(self, max_bytes: int, granularity: int = 2 << 20) -> None:
        self._lib = None
        self.device_index = 0
        self._freed = False
        self.granularity = granularity
        self.reserved_size = vmm_allocator._round_up(max_bytes, granularity)
        self.base_ptr = 0x1000
        self._mapped_granules = set()
        self._handles = []
        self.map_calls = []
        self.fail_on = None

    def _map_granule(self, granule_index: int) -> None:
        if self.fail_on is not None and granule_index == self.fail_on:
            raise VmmUnavailableError("injected mapping failure")
        self.map_calls.append(granule_index)
        self._mapped_granules.add(granule_index)
        self._handles.append((granule_index, self.base_ptr, self.granularity))

    def _unmap_granule(self, granule_index: int) -> None:
        self._mapped_granules.discard(granule_index)
        self._handles = [h for h in self._handles if h[0] != granule_index]

    def free(self) -> None:
        self._freed = True
        self._handles.clear()
        self._mapped_granules.clear()
        self.base_ptr = 0


def test_unavailable_reason_is_string_or_none():
    reason = vmm_unavailable_reason()
    assert reason is None or isinstance(reason, str)
    assert is_vmm_available() == (reason is None)


def test_disable_env_var_forces_unavailable(monkeypatch):
    monkeypatch.setenv("SGLANG_DISABLE_VMM", "1")
    assert vmm_unavailable_reason() == "SGLANG_DISABLE_VMM=1"
    assert not is_vmm_available()


def test_commit_is_monotonic_and_bounded():
    buf = _FakeBuffer(max_bytes=8 << 20)
    assert buf.committed_bytes == 0
    assert buf.commit(1) == buf.granularity
    assert buf.commit(1) == buf.granularity  # idempotent
    assert buf.commit(3 << 20) == 2 * buf.granularity
    assert buf.commit(0) == 2 * buf.granularity  # never shrinks
    assert buf.committed_bytes <= buf.reserved_size


def test_commit_rejects_oversized_request():
    buf = _FakeBuffer(max_bytes=4 << 20)
    with pytest.raises(ValueError):
        buf.commit(buf.reserved_size + 1)
    with pytest.raises(ValueError):
        buf.commit(-1)


def test_commit_rolls_back_on_failure():
    buf = _FakeBuffer(max_bytes=8 << 20)
    buf.commit(1)
    before = buf.committed_bytes
    buf.fail_on = 2
    with pytest.raises(VmmUnavailableError):
        buf.commit(4 << 20)
    assert buf.committed_bytes == before
    assert 2 not in buf._mapped_granules


def test_commit_range_validates_and_maps_only_missing():
    buf = _FakeBuffer(max_bytes=8 << 20)
    buf.commit_range(0, 2 << 20)
    assert buf.committed_bytes == buf.granularity
    buf.commit_range(0, 4 << 20)
    assert buf.committed_bytes == 2 * buf.granularity
    assert buf.map_calls == [0, 1]
    with pytest.raises(ValueError):
        buf.commit_range(2, 1)
    with pytest.raises(ValueError):
        buf.commit_range(0, buf.reserved_size + 1)


def test_free_is_idempotent_and_blocks_commit():
    buf = _FakeBuffer(max_bytes=4 << 20)
    buf.commit(1)
    buf.free()
    assert buf.is_freed
    buf.free()
    with pytest.raises(RuntimeError):
        buf.commit(1)
