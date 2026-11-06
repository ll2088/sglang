# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU-runnable tests for `copy_weight_` in `sglang.srt.layers.parameter`.

The integrated-GPU prefault path cannot run on CI hardware, so the gate is
monkeypatched on and the copy result is checked byte for byte.
"""

import pytest
import torch

from sglang.srt.layers import parameter as parameter_mod
from sglang.srt.layers.parameter import copy_weight_


@pytest.fixture
def prefault_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take the integrated-GPU path of `copy_weight_` without a GPU."""
    monkeypatch.setattr(parameter_mod, "_is_integrated_gpu", lambda device: True)


@pytest.fixture
def prefault_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the plain-copy fallback path."""
    monkeypatch.setattr(parameter_mod, "_is_integrated_gpu", lambda device: False)


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
def test_copy_weight_fallback_copies_exact_bytes(
    prefault_off: None, src: torch.Tensor
) -> None:
    dst = torch.empty_like(src, memory_format=torch.contiguous_format)
    copy_weight_(dst, src)
    expected = src.clone(memory_format=torch.contiguous_format)
    assert torch.equal(
        dst.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)
    )


def test_copy_weight_returns_dst(prefault_on: None) -> None:
    src = torch.randn(8, 8)
    dst = torch.empty_like(src)
    out = copy_weight_(dst, src)
    assert out is dst


def test_copy_weight_skips_prefault_for_non_cpu_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-CPU source must not take the touch path even when the gate is on."""
    calls = []
    monkeypatch.setattr(
        parameter_mod, "_is_integrated_gpu", lambda device: calls.append(device) or True
    )
    src = torch.randn(8, 8)
    dst = torch.empty_like(src)
    copy_weight_(dst, src)
    assert calls == []
    assert torch.equal(dst, src)


def test_copy_weight_skips_prefault_for_dtype_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        parameter_mod, "_is_integrated_gpu", lambda device: calls.append(device) or True
    )
    src = torch.randn(8, 8)
    dst = torch.empty(8, 8, dtype=torch.bfloat16)
    copy_weight_(dst, src)
    assert calls == []
    assert torch.equal(dst, src.to(torch.bfloat16))


def test_copy_weight_skips_prefault_for_non_contiguous_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        parameter_mod, "_is_integrated_gpu", lambda device: calls.append(device) or True
    )
    src = torch.randn(64, 64).t()
    dst = torch.empty_like(src, memory_format=torch.contiguous_format)
    copy_weight_(dst, src)
    assert calls == []
    assert torch.equal(dst, src.contiguous())


def test_copy_weight_prefault_touches_every_page(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The touch must cover the first byte of every page plus the last byte."""
    touched = []
    real_sum = torch.Tensor.sum

    def recording_sum(self, *args, **kwargs):
        touched.append(self.numel())
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", recording_sum)
    src = torch.randn(1 << 20)
    dst = torch.empty_like(src)
    copy_weight_(dst, src)
    assert len(touched) == 2
    assert torch.equal(dst, src)


def test_copy_with_check_uses_copy_weight(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same-dtype copies in `copy_with_check` go through `copy_weight_`."""
    seen = []
    real_copy_weight = parameter_mod.copy_weight_

    def recording_copy_weight(dst, src):
        seen.append((dst.shape, src.shape))
        return real_copy_weight(dst, src)

    monkeypatch.setattr(parameter_mod, "copy_weight_", recording_copy_weight)
    target = torch.empty(4, 4)
    loaded = torch.randn(4, 4)
    parameter_mod.copy_with_check(target, loaded)
    assert seen == [((4, 4), (4, 4))]
    assert torch.equal(target, loaded)
