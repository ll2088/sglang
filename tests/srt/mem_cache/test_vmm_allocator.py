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

from sglang.srt.mem_cache.vmm_allocator import (
    VmmBuffer,
    VmmUnavailableError,
    is_vmm_available,
    vmm_unavailable_reason,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _require_vmm():
    # VMM needs a CUDA/HIP driver exposing the cuMem*/hipMem* API. Hosts
    # without a GPU, or with a driver that lacks the symbols, skip instead
    # of failing so the suite stays meaningful on CPU-only CI runners.
    reason = vmm_unavailable_reason()
    if reason is not None:
        pytest.skip(f"VMM unavailable: {reason}")


@requires_cuda
def test_vmm_availability_is_consistent():
    reason = vmm_unavailable_reason()
    assert is_vmm_available() == (reason is None)


@requires_cuda
def test_commit_never_exceeds_reservation():
    _require_vmm()
    buf = VmmBuffer(8 << 20, torch.cuda.current_device())
    try:
        assert buf.committed_bytes == 0
        committed = buf.commit(3 << 20)
        assert committed <= buf.reserved_size
        assert committed == buf.committed_bytes
        with pytest.raises(ValueError):
            buf.commit(buf.reserved_size + 1)
    finally:
        buf.free()


@requires_cuda
def test_commit_is_idempotent_and_monotonic():
    _require_vmm()
    buf = VmmBuffer(8 << 20, torch.cuda.current_device())
    try:
        first = buf.commit(4 << 20)
        second = buf.commit(4 << 20)
        assert first == second
        smaller = buf.commit(1 << 20)
        assert smaller == first
    finally:
        buf.free()


@requires_cuda
def test_free_releases_reservation():
    _require_vmm()
    buf = VmmBuffer(4 << 20, torch.cuda.current_device())
    buf.commit(2 << 20)
    buf.free()
    assert buf.base_ptr == 0
    with pytest.raises(RuntimeError):
        buf.commit(1 << 20)


def test_unavailable_reason_is_string_or_none():
    reason = vmm_unavailable_reason()
    assert reason is None or isinstance(reason, str)
    if not torch.cuda.is_available():
        assert reason is not None


def test_vmm_unavailable_error_is_runtime_error():
    assert issubclass(VmmUnavailableError, RuntimeError)


def test_commit_range_rejects_invalid_range():
    # Range validation happens before any driver call, so this runs on CPU.
    buf = VmmBuffer.__new__(VmmBuffer)
    buf._freed = False
    buf.reserved_size = 8 << 20
    buf.granularity = 2 << 20
    buf._mapped_granules = set()
    buf._handles = []
    with pytest.raises(ValueError):
        buf.commit_range(1, 0)
    with pytest.raises(ValueError):
        buf.commit_range(0, buf.reserved_size + 1)


def test_commit_range_after_free_raises():
    buf = VmmBuffer.__new__(VmmBuffer)
    buf._freed = True
    buf.reserved_size = 8 << 20
    buf.granularity = 2 << 20
    buf._mapped_granules = set()
    buf._handles = []
    with pytest.raises(RuntimeError):
        buf.commit_range(0, 1)


@requires_cuda
def test_as_tensor_view_stays_valid_across_commit():
    _require_vmm()
    buf = VmmBuffer(8 << 20, torch.cuda.current_device())
    try:
        view = buf.as_tensor(torch.float16, (1024,))
        assert view.data_ptr() == buf.base_ptr
        buf.commit(4 << 20)
        # The view keeps the same base pointer after more pages are mapped.
        assert view.data_ptr() == buf.base_ptr
        assert buf.committed_bytes >= 4 << 20
    finally:
        buf.free()
