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
"""Unit tests for the minimal VMM-backed growable buffer."""

import pytest
import torch

from sglang.srt.mem_cache.vmm_allocator import (
    VmmBuffer,
    VmmUnavailableError,
    is_vmm_available,
    vmm_unavailable_reason,
)

requires_vmm = pytest.mark.skipif(
    not is_vmm_available(),
    reason=f"VMM unavailable: {vmm_unavailable_reason()}",
)


@requires_vmm
def test_reserve_and_commit_prefix():
    buf = VmmBuffer(8 << 20, torch.cuda.current_device())
    try:
        assert buf.committed_bytes == 0
        assert buf.reserved_size >= 8 << 20
        committed = buf.commit(3 << 20)
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
        second = buf.commit(2 << 20)
        assert first == second
        third = buf.commit(1 << 20)
        assert third == first
        fourth = buf.commit(4 << 20)
        assert fourth >= first
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
def test_commit_range_and_tensor_view():
    buf = VmmBuffer(8 << 20, torch.cuda.current_device())
    try:
        buf.commit_range(0, 2 << 20)
        tensor = buf.as_tensor(torch.uint8, (2 << 20,))
        assert tensor.numel() == 2 << 20
        assert tensor.data_ptr() == buf.base_ptr
    finally:
        buf.free()


def test_unavailable_reason_is_string_or_none():
    reason = vmm_unavailable_reason()
    assert reason is None or isinstance(reason, str)


def test_buffer_rejects_when_unavailable():
    if is_vmm_available():
        pytest.skip("VMM is available on this host")
    with pytest.raises(VmmUnavailableError):
        VmmBuffer(1 << 20, 0)
