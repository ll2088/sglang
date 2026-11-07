"""Minimal benchmark entry for the DSA top-k dtype negotiation.

This is a plain script (not a pytest module) so it can run in the minimal
pytest+numpy job without torch.  It measures the host-side negotiation cost of
the BF16 native path versus the FP32 fallback and prints the median latency of
each, which is what the performance gate compares.

Usage:
    python tests/kernels/benchmark_topk_bf16.py
"""

from __future__ import annotations

import statistics
import time

from sglang.kernels.jit.csrc.dsa import topk_bf16_capability as cap

BF16 = (cap.DL_BFLOAT, 16, 1)
FP32 = (cap.DL_FLOAT, 32, 1)


def _time(fn, iters: int = 20000) -> float:
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        samples.append((time.perf_counter() - start) / iters)
    return statistics.median(samples)


def main() -> None:
    bf16 = _time(lambda: cap.resolve_topk_dtype("cooperative", *BF16, 65536))
    fp32 = _time(lambda: cap.resolve_topk_dtype("cooperative", *FP32, 65536))
    print(f"bf16_native_median_s={bf16:.9f}")
    print(f"fp32_fallback_median_s={fp32:.9f}")
    if fp32 > 0:
        print(f"ratio={bf16 / fp32:.3f}")


if __name__ == "__main__":
    main()
