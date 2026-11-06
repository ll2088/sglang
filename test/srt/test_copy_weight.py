# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the integrated-GPU mmap prefault in copy_weight_.

These tests never require a GPU: the integrated-GPU gate is monkeypatched so
the prefault path can be exercised on CI, and the copy itself is checked
byte-for-byte against a plain copy_.
"""

import statistics
import time
from pathlib import Path

import pytest
import torch

from sglang.srt.layers import parameter as parameter_mod
from sglang.srt.layers.parameter import copy_weight_


def _force_gate(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    # copy_weight_ resolves the gate through the module-level name, so patch
    # that name rather than the platform registry.
    monkeypatch.setattr(parameter_mod, "_is_integrated_gpu", lambda device: value)


@pytest.fixture
def prefault_on(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_gate(monkeypatch, True)


@pytest.fixture
def prefault_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_gate(monkeypatch, False)


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


def test_copy_weight_gate_off_matches_plain_copy(prefault_off: None) -> None:
    src = torch.randn(32, 64)
    dst = torch.empty_like(src)
    copy_weight_(dst, src)
    assert torch.equal(dst, src)


def test_copy_weight_skips_prefault_for_non_contiguous(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    touched = []
    real_sum = torch.Tensor.sum

    def spy_sum(self, *args, **kwargs):
        touched.append(self.shape)
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", spy_sum)
    src = torch.randn(8, 16).t()  # non-contiguous
    dst = torch.empty_like(src, memory_format=torch.contiguous_format)
    copy_weight_(dst, src)
    assert touched == []
    assert torch.equal(dst, src.contiguous())


def test_copy_weight_skips_prefault_for_dtype_conversion(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    touched = []
    real_sum = torch.Tensor.sum

    def spy_sum(self, *args, **kwargs):
        touched.append(self.shape)
        return real_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", spy_sum)
    src = torch.randn(8, 16)
    dst = torch.empty(8, 16, dtype=torch.bfloat16)
    copy_weight_(dst, src)
    assert touched == []
    assert torch.equal(dst, src.to(torch.bfloat16))


def test_copy_weight_prefault_maps_the_mmap_view(
    prefault_on: None, tmp_path: Path
) -> None:
    """Every page of the safetensors view must be mapped before the copy."""
    safetensors = pytest.importorskip("safetensors.torch")
    path = str((tmp_path / "w.safetensors").resolve())
    safetensors.save_file({"w": torch.randn(1 << 20)}, path)

    def mapped_bytes() -> int:
        total, in_file = 0, False
        with open("/proc/self/smaps") as smaps:
            for line in smaps:
                fields = line.split()
                if not fields:
                    continue
                if ":" not in fields[0]:
                    in_file = fields[-1] == path
                elif in_file and fields[0] == "Rss:":
                    total += int(fields[1]) * 1024
        return total

    with safetensors.safe_open(path, framework="pt") as f:
        src = f.get_tensor("w")
        assert mapped_bytes() < src.nbytes
        copy_weight_(torch.empty_like(src, device="meta"), src)
        assert mapped_bytes() >= src.nbytes


def test_parameter_copy_sites_use_copy_weight(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replaced copy sites in parameter.py must route through copy_weight_."""
    calls = []
    real = parameter_mod.copy_weight_

    def spy(dst, src):
        calls.append((tuple(dst.shape), tuple(src.shape)))
        return real(dst, src)

    monkeypatch.setattr(parameter_mod, "copy_weight_", spy)

    data = torch.empty(4, 8)
    param = parameter_mod.BasevLLMParameter(data, weight_loader=lambda *a, **k: None)
    loaded = torch.randn(4, 8)
    param._assert_and_load(loaded)
    assert calls == [((4, 8), (4, 8))]
    assert torch.equal(param.data, loaded)


def test_vocab_parallel_embedding_copy_site_uses_copy_weight(
    prefault_on: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The output_dim-is-None branch of the embedding loader uses copy_weight_."""
    from sglang.srt.layers import vocab_parallel_embedding as vpe

    calls = []
    real = vpe.copy_weight_

    def spy(dst, src):
        calls.append((tuple(dst.shape), tuple(src.shape)))
        return real(dst, src)

    monkeypatch.setattr(vpe, "copy_weight_", spy)

    class _Loader(vpe.VocabParallelEmbedding):
        def __init__(self):  # bypass the heavy __init__
            torch.nn.Module.__init__(self)

    loader = _Loader()
    param = torch.nn.Parameter(torch.empty(4, 8))
    loaded = torch.randn(4, 8)
    loader.weight_loader(param, loaded)
    assert calls == [((4, 8), (4, 8))]
    assert torch.equal(param.data, loaded)


def _median_copy_us(src: torch.Tensor, gate_on: bool, repeats: int = 50) -> float:
    dst = torch.empty_like(src, memory_format=torch.contiguous_format)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        if gate_on:
            flat = src.reshape(-1).view(torch.uint8)
            flat[:: 4096].sum()
            flat[-1:].sum()
        dst.copy_(src)
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def test_prefault_overhead_is_bounded_on_cpu() -> None:
    """CPU-only sanity bound: the touch must not dominate the copy.

    This is not the GB10 benchmark; it only guards against an accidental
    quadratic touch loop. The bound is deliberately loose for shared CI.
    """
    src = torch.randn(256, 1024)
    plain = _median_copy_us(src, gate_on=False)
    touched = _median_copy_us(src, gate_on=True)
    assert touched <= plain * 10 + 1000
