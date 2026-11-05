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
"""Tests for the minimal VMM-backed growable device buffer.

Driver-dependent tests are skipped when VMM is unavailable so the suite runs
on CPU-only CI hosts; the pure helpers are always exercised.
"""

import pytest
import torch

from sglang.srt.mem_cache.vmm_allocator import (
    VmmBuffer,
    VmmUnavailableError,
    is_vmm_available,
    vmm_unavailable_reason,
)

requires_vmm = pytest.mark.skipif(
    not is_vmm_available(), reason=f"VMM unavailable: {vmm_unavailable_reason()}"
)


def test_unavailable_reason_is_string_or_none():
    reason = vmm_unavailable_reason()
    assert reason is None or isinstance(reason, str)


def test_is_vmm_available_matches_reason():
    assert is_vmm_available() == (vmm_unavailable_reason() is None)


def test_buffer_rejects_non_positive_size():
    if not is_vmm_available():
        pytest.skip("VMM unavailable")
    with pytest.raises(ValueError):
        VmmBuffer(0, 0)


@requires_vmm
def test_reserve_then_commit_prefix():
    buf = VmmBuffer(8 << 20, torch.cuda.current_device())
    try:
        assert buf.committed_bytes == 0
        assert buf.reserved_size >= 8 << 20
        committed = buf.commit(4 << 20)
        assert committed > 0
        assert committed <= buf.reserved_size
        assert buf.committed_bytes == committed
    finally:
        buf.free()


@requires_vmm
def test_commit_is_idempotent_and_monotonic():
    buf = VmmBuffer(8 << 20, torch.cuda.current_device())
    try:
        first = buf.commit(2 << 20)
        assert buf.commit(2 << 20) == first
        assert buf.commit(1 << 20) == first
        second = buf.commit(6 << 20)
        assert second >= first
    finally:
        buf.free()


@requires_vmm
def test_commit_rejects_oversized_request():
    buf = VmmBuffer(4 << 20, torch.cuda.current_device())
    try:
        with pytest.raises(ValueError):
            buf.commit(buf.reserved_size + 1)
    finally:
        buf.free()


@requires_vmm
def test_commit_range_validates_bounds():
    buf = VmmBuffer(4 << 20, torch.cuda.current_device())
    try:
        with pytest.raises(ValueError):
            buf.commit_range(1, 0)
        with pytest.raises(ValueError):
            buf.commit_range(0, buf.reserved_size + 1)
    finally:
        buf.free()


@requires_vmm
def test_free_is_idempotent():
    buf = VmmBuffer(2 << 20, torch.cuda.current_device())
    buf.free()
    buf.free()
    assert buf.is_freed


@requires_vmm
def test_commit_after_free_raises():
    buf = VmmBuffer(2 << 20, torch.cuda.current_device())
    buf.free()
    with pytest.raises(RuntimeError):
        buf.commit(1 << 20)


@requires_vmm
def test_tensor_view_stays_valid_across_commits():
    buf = VmmBuffer(4 << 20, torch.cuda.current_device())
    try:
        view = buf.as_tensor(torch.uint8, (buf.reserved_size,))
        assert view.data_ptr() == buf.base_ptr
        buf.commit(1 << 20)
        assert view.data_ptr() == buf.base_ptr
    finally:
        buf.free()
