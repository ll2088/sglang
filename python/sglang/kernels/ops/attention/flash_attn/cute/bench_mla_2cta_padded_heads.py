# Copyright (c) 2025, Tri Dao.
"""Benchmark harness for dense 2CTA MLA with padded Q heads (65-127).

This script is intentionally not a pytest test: it needs a B200-class GPU and
the full sglang install. It exits with a clear skip message when those are
unavailable so CI can call it without failing the environment setup.

Usage:
    python bench_mla_2cta_padded_heads.py --heads 96 --baseline-heads 128

It reports median latency over repeated CUDA-graph replays for the requested
head count and the 128-head baseline, and fails only if the padded path is more
than 10% slower than the baseline on the same shape.
"""

import argparse
import statistics
import sys


def _skip(reason):
    print(f"SKIP: {reason}")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--baseline-heads", type=int, default=128)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seqlen-k", type=int, default=256)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--max-slowdown", type=float, default=0.10)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        _skip("torch is not installed")
    if not torch.cuda.is_available():
        _skip("CUDA is not available")
    try:
        from sglang.kernels.ops.attention.flash_attn.cute import flash_attn_func
    except ImportError as exc:
        _skip(f"sglang flash-attn cute is not importable: {exc}")

    from sglang.kernels.ops.attention.flash_attn.cute.mla_head_padding import (
        mla_2cta_pads_dense_qheads,
    )

    if not mla_2cta_pads_dense_qheads(args.heads, 1):
        _skip(f"{args.heads} heads is not on the dense 2CTA padding path")

    device, dtype = "cuda", torch.bfloat16
    head_dim, head_dim_v = 64, 512

    def make_inputs(heads):
        q = torch.randn(args.batch, 1, heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(args.batch, args.seqlen_k, 1, head_dim, device=device, dtype=dtype)
        v = torch.randn(args.batch, args.seqlen_k, 1, head_dim_v, device=device, dtype=dtype)
        return q, k, v

    def time_heads(heads):
        q, k, v = make_inputs(heads)
        for _ in range(5):
            flash_attn_func(q, k, v, causal=False)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            flash_attn_func(q, k, v, causal=False)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        return statistics.median(samples)

    padded_ms = time_heads(args.heads)
    baseline_ms = time_heads(args.baseline_heads)
    slowdown = (padded_ms - baseline_ms) / baseline_ms
    print(
        f"heads={args.heads} median={padded_ms:.4f} ms, "
        f"baseline_heads={args.baseline_heads} median={baseline_ms:.4f} ms, "
        f"slowdown={slowdown:.2%}"
    )
    if slowdown > args.max_slowdown:
        print(
            f"FAIL: padded path is {slowdown:.2%} slower than baseline "
            f"(limit {args.max_slowdown:.2%})"
        )
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
