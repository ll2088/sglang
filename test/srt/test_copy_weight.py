# SPDX-License-Identifier: Apache-2.0
"""CPU-runnable tests for `copy_weight_` (integrated-GPU mmap prefault).

CI has no integrated GPU, so the gate is forced on with monkeypatch. The
prefault path must accept every shape/dtype the weight loaders pass and leave
the copy byte-identical to a plain `copy_`.
"""

import pytest
import torch

from sglang.srt.layers.parameter import copy_weight_


@pytest.fixture
def prefault_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take the integrated-GPU path of `copy_weight_` without a GPU."""
    monkeypatch.setattr(
        "sglang.srt.layers.parameter._is_integrated_gpu", lambda device: True
    )


@pytest.fixture
def prefault_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the plain-copy path regardless of the host."""
    monkeypatch.setattr(
        "sglang.srt.layers.parameter._is_integrated_gpu", lambda device: False
    )


@pytest.mark.parametrize(
    "src",
    [
        torch.tensor(0.5),
        torch.empty(0, 8),
        torch.randn(64, 1024).to(torch.bfloat16),
        torch.randint(0, 256, (64, 1024), dtype=torch.uint8).view(torch.float8_e4m3fn),
        torch.randn(64, 1024).narrow(1, 512, 512),
        torch.arange(2, dtype=torch.float32)[::2],
    ],
    ids=["0d", "empty", "bf16", "fp8", "dim1_narrow", "strided_1elem"],
)
def test_copy_weight_prefault_copies_exact_bytes(
    prefault_on: None, src: torch.Tensor
) -> None:
    dst = torch.empty_like(src, memory_format=torch.contiguous_format)
    copy_weight_(dst, src)
    expected = src.clone(memory_format=torch.contiguous_format)
    assert torch.equal(
        dst.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)
    )


@pytest.mark.parametrize(
    "src",
    [
        torch.tensor(0.5),
        torch.empty(0, 8),
        torch.randn(64, 1024).to(torch.bfloat16),
        torch.randn(64, 1024).narrow(1, 512, 512),
    ],
    ids=["0d", "empty", "bf16", "dim1_narrow"],
)
def test_copy_weight_plain_path_copies_exact_bytes(
    prefault_off: None, src: torch.Tensor
) -> None:
    dst = torch.empty_like(src, memory_format=torch.contiguous_format)
    copy_weight_(dst, src)
    expected = src.clone(memory_format=torch.contiguous_format)
    assert torch.equal(
        dst.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)
    )


def test_copy_weight_prefault_touches_contiguous_source(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prefault path must read the source before the copy."""
    src = torch.randn(64, 1024)
    dst = torch.empty_like(src)
    touched = {"n": 0}
    real_sum = torch.Tensor.sum

    def counting_sum(self, *args, **kwargs):
        touched["n"] += 1
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", counting_sum)
    copy_weight_(dst, src)
    assert touched["n"] >= 1
    assert torch.equal(dst, src)


def test_copy_weight_plain_path_does_not_touch(
    prefault_off: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = torch.randn(64, 1024)
    dst = torch.empty_like(src)
    touched = {"n": 0}
    real_sum = torch.Tensor.sum

    def counting_sum(self, *args, **kwargs):
        touched["n"] += 1
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", counting_sum)
    copy_weight_(dst, src)
    assert touched["n"] == 0
    assert torch.equal(dst, src)


def test_copy_weight_skips_non_contiguous_source(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-contiguous sources are staged by PyTorch; no prefault needed."""
    src = torch.randn(64, 1024).t()
    dst = torch.empty_like(src, memory_format=torch.contiguous_format)
    touched = {"n": 0}
    real_sum = torch.Tensor.sum

    def counting_sum(self, *args, **kwargs):
        touched["n"] += 1
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", counting_sum)
    copy_weight_(dst, src)
    assert touched["n"] == 0
    assert torch.equal(dst, src.contiguous())


def test_copy_weight_skips_dtype_converting_source(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = torch.randn(64, 1024)
    dst = torch.empty(64, 1024, dtype=torch.bfloat16)
    touched = {"n": 0}
    real_sum = torch.Tensor.sum

    def counting_sum(self, *args, **kwargs):
        touched["n"] += 1
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", counting_sum)
    copy_weight_(dst, src)
    assert touched["n"] == 0
    assert torch.equal(dst, src.to(torch.bfloat16))
