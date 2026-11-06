"""Microbenchmark for the Triton prefill attention kernel on gfx1151.

Records the median latency of ``context_attention_fwd`` for the shapes the
split-D head-dim blocking targets (head_dim=72 -> 64 + 16) so a change can be
compared against the pre-change baseline. The kernel itself is untouched; the
split-D path is selected by the existing ``_on_gfx1151`` guard.

Run:
    python benchmark/kernels/attention/bench_prefill_attention_gfx1151.py

Output: one JSON line per shape to stdout.
"""

from __future__ import annotations

import argparse
import json

import torch

from sglang.kernels.ops.attention.prefill_attention import context_attention_fwd


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=72)
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def main() -> int:
    args = _parse()
    if not torch.cuda.is_available():
        raise SystemExit("requires a CUDA/HIP device")

    dtype = _dtype(args.dtype)
    device = "cuda"
    total = args.batch * args.seq

    q = torch.randn(total, args.heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(total, args.kv_heads, args.head_dim, device=device, dtype=dtype)
    v = torch.randn(total, args.kv_heads, args.head_dim, device=device, dtype=dtype)
    o = torch.empty_like(q)
    b_start_loc = torch.arange(
        0, total, args.seq, device=device, dtype=torch.int32
    )
    b_seq_len = torch.full((args.batch,), args.seq, device=device, dtype=torch.int32)
    sm_scale = 1.0 / (args.head_dim**0.5)

    def run() -> None:
        context_attention_fwd(
            q,
            k,
            v,
            o,
            b_start_loc,
            b_seq_len,
            args.seq,
            is_causal=False,
            sm_scale=sm_scale,
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.rep):
        run()
    end.record()
    torch.cuda.synchronize()

    median_ms = start.elapsed_time(end) / args.rep
    print(
        json.dumps(
            {
                "batch": args.batch,
                "seq": args.seq,
                "heads": args.heads,
                "kv_heads": args.kv_heads,
                "head_dim": args.head_dim,
                "dtype": args.dtype,
                "median_ms": median_ms,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
