# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU-only tests for the integrated-GPU mmap prefault in `copy_weight_`.

The prefault path only runs on integrated GPUs, which CI does not have, so the
platform gate is monkeypatched on. The copy itself must stay byte-exact.
"""

import mmap

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
    """Force the plain-copy path regardless of the host hardware."""
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
def test_copy_weight_plain_path_copies_exact_bytes(
    prefault_off: None, src: torch.Tensor
) -> None:
    dst = torch.empty_like(src, memory_format=torch.contiguous_format)
    copy_weight_(dst, src)
    expected = src.clone(memory_format=torch.contiguous_format)
    assert torch.equal(
        dst.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)
    )


def test_copy_weight_prefault_touches_every_page(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prefault must touch one byte per page plus the last byte."""
    src = torch.randn(4 * mmap.PAGESIZE, dtype=torch.uint8)
    touched: list[int] = []
    real_sum = torch.Tensor.sum

    def spy_sum(self, *args, **kwargs):
        touched.append(self.numel())
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", spy_sum)
    dst = torch.empty_like(src)
    copy_weight_(dst, src)
    monkeypatch.undo()

    # One strided touch (one byte per page) plus the trailing byte.
    assert touched == [4, 1]
    assert torch.equal(dst, src)


def test_copy_weight_skips_prefault_for_non_cpu_source(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-CPU source must not take the touch path."""
    calls: list[int] = []
    real_sum = torch.Tensor.sum

    def spy_sum(self, *args, **kwargs):
        calls.append(self.numel())
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", spy_sum)
    src = torch.randn(8, 8)
    dst = torch.empty_like(src)
    # Pretend the source lives on a device; the gate must reject it.
    monkeypatch.setattr(
        torch.Tensor, "device", property(lambda self: torch.device("meta"))
    )
    copy_weight_(dst, src)
    monkeypatch.undo()
    assert calls == []


def test_copy_weight_skips_prefault_for_dtype_conversion(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dtype-converting copies are staged by PyTorch; no touch needed."""
    calls: list[int] = []
    real_sum = torch.Tensor.sum

    def spy_sum(self, *args, **kwargs):
        calls.append(self.numel())
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", spy_sum)
    src = torch.randn(8, 8)
    dst = torch.empty(8, 8, dtype=torch.bfloat16)
    copy_weight_(dst, src)
    monkeypatch.undo()
    assert calls == []
    assert torch.equal(dst, src.to(torch.bfloat16))


def test_vocab_parallel_embedding_loader_matches_plain_copy() -> None:
    """The embedding loader's copy point must stay numerically identical."""
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    num_embeddings, embedding_dim = 128, 16
    layer = VocabParallelEmbedding(
        num_embeddings,
        embedding_dim,
        params_dtype=torch.float32,
        enable_tp=False,
    )
    loaded = torch.randn(num_embeddings, embedding_dim)
    layer.weight_loader(layer.weight, loaded)
    assert torch.equal(layer.weight.data, loaded)
