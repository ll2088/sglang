# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark: float32 vs float64 SVD calibration path."""

import statistics
import time

import torch

from sglang.multimodal_gen.runtime.layers.quantization.svd_utils import svd


def _time_once(weight: torch.Tensor, rank: int) -> float:
    start = time.perf_counter()
    svd(weight, rank)
    return time.perf_counter() - start


def main() -> None:
    torch.manual_seed(0)
    rank = 32
    w32 = torch.randn(512, 256, dtype=torch.float32)
    w64 = w32.to(torch.float64)
    t32 = statistics.median(_time_once(w32, rank) for _ in range(3))
    t64 = statistics.median(_time_once(w64, rank) for _ in range(3))
    print(f"float32: {t32:.4f}s  float64: {t64:.4f}s")
    assert t32 <= t64, "float32 SVD should not be slower than float64"


if __name__ == "__main__":
    main()
