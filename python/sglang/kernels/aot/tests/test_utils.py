import sys
from unittest import mock

import pytest
import torch

from utils import _peak_gpu_mem_increase


def test_peak_gpu_mem_increase_uses_allocator_stats_without_cuda():
    """Verify the helper logic on CPU-only CI by mocking the CUDA allocator API."""
    device = 0
    before = 1000
    peak = 5000

    def _fn():
        return None

    with mock.patch.object(torch.cuda, "synchronize") as sync, mock.patch.object(
        torch.cuda, "reset_peak_memory_stats"
    ) as reset, mock.patch.object(
        torch.cuda, "memory_allocated", return_value=before
    ) as allocated, mock.patch.object(
        torch.cuda, "max_memory_allocated", return_value=peak
    ) as max_allocated:
        mem_increase = _peak_gpu_mem_increase(_fn, device=device)

    assert mem_increase == peak - before
    assert sync.call_count == 2
    reset.assert_called_once_with(device)
    allocated.assert_called_once_with(device)
    max_allocated.assert_called_once_with(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_peak_gpu_mem_increase_counts_allocations():
    n_bytes = 4 * 1024 * 1024
    device = torch.cuda.current_device()

    def _allocate():
        return torch.empty(n_bytes, dtype=torch.uint8, device=device)

    # Warm up the allocator so the measurement is not polluted by first-use overhead.
    _allocate()
    torch.cuda.synchronize(device)

    mem_increase = _peak_gpu_mem_increase(_allocate, device=device)
    assert mem_increase >= n_bytes
    assert mem_increase < n_bytes * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_peak_gpu_mem_increase_noop_is_zero():
    device = torch.cuda.current_device()

    def _noop():
        return None

    assert _peak_gpu_mem_increase(_noop, device=device) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_peak_gpu_mem_increase_counts_freed_temporaries():
    n_bytes = 4 * 1024 * 1024
    device = torch.cuda.current_device()

    def _allocate_and_free():
        tmp = torch.empty(n_bytes, dtype=torch.uint8, device=device)
        del tmp

    # Warm up the allocator so the measurement is not polluted by first-use overhead.
    torch.empty(n_bytes, dtype=torch.uint8, device=device)
    torch.cuda.synchronize(device)

    mem_increase = _peak_gpu_mem_increase(_allocate_and_free, device=device)
    assert mem_increase >= n_bytes
    assert mem_increase < n_bytes * 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
