"""Microbenchmark: fused RMSNorm + per-token FP8 quant vs the unfused baseline
(RMSNorm followed by a separate per-token FP8 quant).

Adapted from NVIDIA/TensorRT-LLM PR #21287 to measure the fused kernel added in
``sglang.kernels.ops.layernorm.fused_rms_norm_fp8_per_token``.

Run:
    python benchmark/kernels/bench_fused_rmsnorm_per_token_fp8_quant.py
"""

import itertools

import numpy as np
import torch
import triton

from sglang.kernels.ops.layernorm.fused_rms_norm_fp8_per_token import (
    add_rms_norm_fp8_per_token,
)
from sglang.kernels.ops.quantization.fp8_kernel import per_token_quant_fp8

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this benchmark")

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN_SIZES = [2560, 9216]
EPS = 1e-6


def make_inputs(num_tokens, hidden_size, add_residual):
    x = torch.randn(num_tokens, hidden_size, device=DEVICE, dtype=DTYPE)
    residual = torch.randn_like(x) if add_residual else None
    weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE) * 0.2
    return x, residual, weight


def run_unfused(x, residual, weight):
    xf = x.float() + residual.float() if residual is not None else x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()).to(
        DTYPE
    )
    return per_token_quant_fp8(y)


def run_fused(x, residual, weight):
    _, q, scale = add_rms_norm_fp8_per_token(x, residual, weight, EPS, False)
    return q, scale


RUNNERS = {"unfused": run_unfused, "fused": run_fused}


def _bench_ms(fn, args):
    times = triton.testing.do_bench(lambda: fn(*args), warmup=25, rep=100)
    return float(times)


def _check_correctness():
    for hidden_size, add_residual in itertools.product(HIDDEN_SIZES, [False, True]):
        x, residual, weight = make_inputs(64, hidden_size, add_residual)
        with torch.inference_mode():
            q_ref, s_ref = run_unfused(
                x.clone(), residual.clone() if add_residual else None, weight
            )
            q, s = run_fused(
                x.clone(), residual.clone() if add_residual else None, weight
            )
        torch.testing.assert_close(s, s_ref, rtol=0, atol=0)
        assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    print("correctness check passed (fused == unfused per-token FP8 quant)")


configs = [
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[1, 50, 512, 1024, 2048, 4096],
        x_log=False,
        line_arg="provider",
        line_vals=list(RUNNERS),
        line_names=list(RUNNERS),
        styles=[("blue", "-"), ("green", "-")],
        ylabel="latency (ms)",
        plot_name=f"rmsnorm_per_token_fp8_quant_h{hidden_size}_residual{add_residual}",
        args={"hidden_size": hidden_size, "add_residual": add_residual},
    )
    for hidden_size, add_residual in itertools.product(HIDDEN_SIZES, [False, True])
]


@triton.testing.perf_report(configs)
def benchmark(num_tokens, hidden_size, add_residual, provider):
    x, residual, weight = make_inputs(num_tokens, hidden_size, add_residual)
    return _bench_ms(RUNNERS[provider], (x, residual, weight))


if __name__ == "__main__":
    torch.manual_seed(0)
    _check_correctness()
    benchmark.run(print_data=True, show_plots=False)
