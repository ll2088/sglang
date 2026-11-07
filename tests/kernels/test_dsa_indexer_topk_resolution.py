"""Pure-Python tests for indexer_topk backend dtype negotiation and fallback.

These tests do not import sglang or torch.  They exercise the host-side
capability predicates that decide whether a backend takes BF16 logits or must
fall back to FP32, mirroring the kernel-side ``takes_bf16_logits`` checks.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest


def _load_capability_module():
    here = pathlib.Path(__file__).resolve()
    repo_root = here.parents[2]
    path = repo_root / "python" / "sglang" / "kernels" / "jit" / "csrc" / "dsa" / "bf16_capability.py"
    if not path.is_file():
        pytest.skip(f"bf16_capability.py not found at {path}")
    spec = importlib.util.spec_from_file_location("dsa_bf16_capability_resolution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cap = _load_capability_module()


@pytest.mark.parametrize("backend", ["cooperative", "persistent", "deep_select"])
@pytest.mark.parametrize("stride", [8, 16, 64, 65536])
def test_bf16_backends_take_aligned_bf16(backend, stride):
    assert cap.resolve_logits_dtype(cap.BF16, stride) == "bf16"


@pytest.mark.parametrize("backend", ["cooperative", "persistent", "deep_select"])
@pytest.mark.parametrize("stride", [1, 2, 3, 4, 7, 9, 15])
def test_bf16_backends_fall_back_on_unaligned(backend, stride):
    assert cap.resolve_logits_dtype(cap.BF16, stride) == "fp32"


def test_fp32_backend_never_takes_bf16():
    assert cap.resolve_logits_dtype(cap.FP32, 8) == "fp32"
    assert not cap.takes_bf16_logits(cap.FP32, 8)


def test_unknown_dtype_falls_back_to_fp32():
    # Any dtype that is not BF16 must resolve to the FP32 path.
    assert cap.resolve_logits_dtype("float16", 8) == "fp32"


def test_fallback_is_reported():
    # The resolver must expose whether a fallback happened so the dispatcher
    # can log it; unaligned BF16 is the canonical fallback case.
    assert cap.resolve_logits_dtype(cap.BF16, 3) == "fp32"
    assert cap.takes_bf16_logits(cap.BF16, 3) is False
