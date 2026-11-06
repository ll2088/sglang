# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for `copy_weight_` and the weight-loader copy sites.

On integrated GPUs `copy_weight_` touches one byte per page of a contiguous,
same-dtype CPU source before the host-to-device copy, so the page faults of a
safetensors mmap happen on the CPU instead of inside the driver copy. CI has no
integrated GPU, so the gate is monkeypatched on.
"""

import mmap

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
    """Force the plain-copy path regardless of the host hardware."""
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
    """The touch must accept the shapes and dtypes weight loaders pass and
    leave the copy unchanged."""
    dst = torch.empty_like(src, memory_format=torch.contiguous_format)
    copy_weight_(dst, src)
    expected = src.clone(memory_format=torch.contiguous_format)
    assert torch.equal(
        dst.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)
    )


def test_copy_weight_prefault_touches_every_page(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One byte per page plus the last byte must be read before the copy."""
    src = torch.randn(4 * mmap.PAGESIZE, dtype=torch.uint8)
    touched = []
    real_sum = torch.Tensor.sum

    def spy_sum(self, *args, **kwargs):
        touched.append(self.numel())
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", spy_sum)
    dst = torch.empty_like(src)
    copy_weight_(dst, src)
    monkeypatch.undo()

    assert touched, "prefault path did not touch the source"
    assert torch.equal(dst, src)


def test_copy_weight_prefault_skips_non_contiguous(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-contiguous sources are staged by PyTorch already; no touch."""
    src = torch.randn(64, 128).t()
    assert not src.is_contiguous()
    touched = []
    real_sum = torch.Tensor.sum

    def spy_sum(self, *args, **kwargs):
        touched.append(self.numel())
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", spy_sum)
    dst = torch.empty_like(src, memory_format=torch.contiguous_format)
    copy_weight_(dst, src)
    monkeypatch.undo()

    assert not touched
    assert torch.equal(dst, src.contiguous())


def test_copy_weight_prefault_skips_dtype_conversion(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dtype-converting copies already go through a CPU temporary."""
    src = torch.randn(64, 128)
    touched = []
    real_sum = torch.Tensor.sum

    def spy_sum(self, *args, **kwargs):
        touched.append(self.numel())
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", spy_sum)
    dst = torch.empty_like(src, dtype=torch.bfloat16)
    copy_weight_(dst, src)
    monkeypatch.undo()

    assert not touched
    assert torch.equal(dst, src.to(torch.bfloat16))


def test_copy_weight_gate_off_is_plain_copy(
    prefault_off: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off integrated GPUs the helper must not touch the source at all."""
    src = torch.randn(64, 1024)
    touched = []
    real_sum = torch.Tensor.sum

    def spy_sum(self, *args, **kwargs):
        touched.append(self.numel())
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", spy_sum)
    dst = torch.empty_like(src)
    copy_weight_(dst, src)
    monkeypatch.undo()

    assert not touched
    assert torch.equal(dst, src)


def test_copy_weight_returns_dst(prefault_on: None) -> None:
    src = torch.randn(8, 8)
    dst = torch.empty_like(src)
    assert copy_weight_(dst, src) is dst
