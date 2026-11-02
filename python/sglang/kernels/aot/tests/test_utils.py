import sys

import pytest
import torch

from utils import _peak_gpu_mem_increase


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_peak_gpu_mem_increase_counts_allocations():
    n_bytes = 4 * 1024 * 1024

    def _allocate():
        return torch.empty(n_bytes, dtype=torch.uint8, device="cuda")

    # Warm up the allocator so the measurement is not polluted by first-use overhead.
    _allocate()
    torch.cuda.synchronize()

    mem_increase = _peak_gpu_mem_increase(_allocate)
    assert mem_increase >= n_bytes
    assert mem_increase < n_bytes * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_peak_gpu_mem_increase_noop_is_zero():
    def _noop():
        return None

    assert _peak_gpu_mem_increase(_noop) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
