"""Tests for the fused per-tensor FP8 quantizer.

Covers numerical correctness against a torch reference, boundary cases
(empty, all-zero, extreme values, non-aligned lengths), and import-time
safety of the attention kernel registry.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.per_tensor_fp8_quant import (
    _BLOCK,
    _MIN_DESCALE,
    _PARTS,
    fused_per_tensor_fp8_quant,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=60, suite="nightly-amd-kernel-1-gpu", nightly=True)

if not torch.cuda.is_available():
    pytest.skip(
        "fused_per_tensor_fp8_quant requires a CUDA/ROCm device",
        allow_module_level=True,
    )

DEVICE = "cuda"
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


def _reference_quant(x: torch.Tensor):
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / FP8_MAX if amax > 0 else amax.new_tensor(_MIN_DESCALE)
    x_fp8 = (x.float() / descale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, descale.reshape(1)


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
@pytest.mark.parametrize("num_inputs", [1, 2, 3])
+def test_fused_per_tensor_fp8_quant_matches_reference(
+    num_tokens: int, num_inputs: int
+) -> None:
+    torch.manual_seed(0)
+    heads, dim = 12, 192
+    inputs = [
+        (torch.randn(num_tokens, heads, dim, device=DEVICE) * (i + 1)).to(
+            torch.bfloat16
+        )
+        for i in range(num_inputs)
+    ]
+    outputs, descales = fused_per_tensor_fp8_quant(*inputs)
+    assert len(outputs) == num_inputs and len(descales) == num_inputs
+    for x, x_fp8, descale in zip(inputs, outputs, descales):
+        ref_fp8, ref_descale = _reference_quant(x)
+        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
+        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
+        if x.numel() == 0:
+            continue
+        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
+        assert ulp_diff.abs().max() <= 1
+
+
+@pytest.mark.parametrize(
+    "numel",
+    [1, _BLOCK, _BLOCK + 1, _PARTS * _BLOCK - 1, _PARTS * _BLOCK, _PARTS * _BLOCK + 1],
+)
+def test_fused_per_tensor_fp8_quant_covers_every_element(numel: int) -> None:
+    x = torch.full((numel, 1, 1), 0.5, device=DEVICE, dtype=torch.bfloat16)
+    expected = 0.5 / FP8_MAX
+    outputs, descales = fused_per_tensor_fp8_quant(x)
+    torch.testing.assert_close(descales[0], torch.tensor([expected], device=DEVICE))
+    assert outputs[0].float().abs().max() <= FP8_MAX
+
+
+def test_fused_per_tensor_fp8_quant_all_zero_and_extreme() -> None:
+    s, heads, dim = 16, 4, 128
+    zeros = torch.zeros(s, heads, dim, device=DEVICE, dtype=torch.bfloat16)
+    outputs, descales = fused_per_tensor_fp8_quant(zeros)
+    assert torch.isfinite(outputs[0].float()).all()
+    assert descales[0].item() == pytest.approx(_MIN_DESCALE)
+
+    extreme = torch.full((s, heads, dim), 1e4, device=DEVICE, dtype=torch.bfloat16)
+    outputs, _ = fused_per_tensor_fp8_quant(extreme)
+    assert outputs[0].float().abs().max() <= FP8_MAX
+
+
+def test_fused_per_tensor_fp8_quant_strided_view() -> None:
+    s, heads = 8, 4
+    kv = torch.randn(s, heads, 128 + 128, device=DEVICE, dtype=torch.bfloat16)
+    v = kv[..., 128:]
+    outputs, descales = fused_per_tensor_fp8_quant(v)
+    ref_fp8, ref_descale = _reference_quant(v)
+    torch.testing.assert_close(descales[0], ref_descale, rtol=1e-6, atol=0)
+    ulp_diff = outputs[0].view(torch.int8).int() - ref_fp8.view(torch.int8).int()
+    assert ulp_diff.abs().max() <= 1
+
+
+def test_fused_per_tensor_fp8_quant_rejects_bad_inputs() -> None:
+    x = torch.zeros(4, 2, 8, device=DEVICE, dtype=torch.bfloat16)
+    with pytest.raises(ValueError):
+        fused_per_tensor_fp8_quant()
+    with pytest.raises(ValueError):
+        fused_per_tensor_fp8_quant(x, x, x, x)
+    with pytest.raises(ValueError):
+        fused_per_tensor_fp8_quant(x.transpose(1, 2))
+
+
+def test_attention_registry_import_smoke() -> None:
+    import sglang.kernels.ops.attention as m  # noqa: F401
+    from sglang.kernels.registry import registry
+    from sglang.kernels.spec import KernelBackend
+
+    spec = registry.get_backend(
+        "attention.fused_per_tensor_fp8_quant", KernelBackend.TRITON
+    )
+    assert spec.target.endswith(":fused_per_tensor_fp8_quant")
+
+
+if __name__ == "__main__":
+    sys.exit(pytest.main([__file__, "-v", "-s"]))
