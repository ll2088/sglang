# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

"""Unit tests for the platform-aware MoE all-to-all backend resolution."""

import pytest

from sglang.srt.distributed.device_communicators import all2all
from sglang.srt.distributed.device_communicators.all2all import (
    DEFAULT_MOE_A2A_BACKEND,
    ONE_SIDED_MOE_A2A_BACKEND,
    clear_platform_probes,
    require_single_node_without_mnnvl,
    resolve_moe_a2a_backend,
)


@pytest.fixture(autouse=True)
def _clear_probes():
    clear_platform_probes()
    yield
    clear_platform_probes()


def _pretend_platform(monkeypatch, *, cuda, arch, capability):
    monkeypatch.setattr(all2all, "is_cuda", lambda: cuda)
    monkeypatch.setattr(all2all, "_cpu_architecture", lambda: arch)
    monkeypatch.setattr(all2all, "_device_capability", lambda: capability)
    clear_platform_probes()


@pytest.mark.parametrize(
    ("arch", "capability"),
    [
        ("aarch64", (10, 0)),  # GB200
        ("aarch64", (10, 3)),  # GB300
        ("aarch64", (10, 7)),  # VR200
    ],
)
def test_auto_enables_one_sided_on_arm_blackwell(monkeypatch, arch, capability):
    _pretend_platform(monkeypatch, cuda=True, arch=arch, capability=capability)
    assert (
        resolve_moe_a2a_backend(
            "auto",
            enable_expert_parallel=True,
            nnodes=2,
            data_parallel_size=8,
            data_parallel_size_local=0,
        )
        == ONE_SIDED_MOE_A2A_BACKEND
    )


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_auto_enables_one_sided_on_single_node_hgx(monkeypatch, capability):
    _pretend_platform(monkeypatch, cuda=True, arch="x86_64", capability=capability)
    assert (
        resolve_moe_a2a_backend(
            "auto",
            enable_expert_parallel=True,
            nnodes=1,
            data_parallel_size=8,
            data_parallel_size_local=8,
        )
        == ONE_SIDED_MOE_A2A_BACKEND
    )


def test_auto_falls_back_on_multi_node_hgx(monkeypatch):
    _pretend_platform(monkeypatch, cuda=True, arch="x86_64", capability=(10, 0))
    assert (
        resolve_moe_a2a_backend(
            "auto",
            enable_expert_parallel=True,
            nnodes=2,
            data_parallel_size=8,
            data_parallel_size_local=8,
        )
        == DEFAULT_MOE_A2A_BACKEND
    )


@pytest.mark.parametrize(
    ("arch", "capability"),
    [
        ("aarch64", (12, 1)),  # GB10
        ("aarch64", (9, 0)),  # GH200
        ("x86_64", (9, 0)),  # H100
    ],
)
def test_auto_falls_back_on_other_platforms(monkeypatch, arch, capability):
    _pretend_platform(monkeypatch, cuda=True, arch=arch, capability=capability)
    assert (
        resolve_moe_a2a_backend(
            "auto",
            enable_expert_parallel=True,
            nnodes=1,
            data_parallel_size=8,
            data_parallel_size_local=8,
        )
        == DEFAULT_MOE_A2A_BACKEND
    )


def test_auto_falls_back_without_expert_parallel(monkeypatch):
    _pretend_platform(monkeypatch, cuda=True, arch="aarch64", capability=(10, 0))
    assert (
        resolve_moe_a2a_backend(
            "auto",
            enable_expert_parallel=False,
            nnodes=1,
            data_parallel_size=1,
            data_parallel_size_local=1,
        )
        == DEFAULT_MOE_A2A_BACKEND
    )


def test_auto_falls_back_with_elastic_ep(monkeypatch):
    _pretend_platform(monkeypatch, cuda=True, arch="aarch64", capability=(10, 0))
    assert (
        resolve_moe_a2a_backend(
            "auto",
            enable_expert_parallel=True,
            enable_elastic_ep=True,
            nnodes=1,
            data_parallel_size=1,
            data_parallel_size_local=1,
        )
        == DEFAULT_MOE_A2A_BACKEND
    )


def test_explicit_backend_overrides_auto(monkeypatch):
    _pretend_platform(monkeypatch, cuda=True, arch="aarch64", capability=(10, 0))
    assert (
        resolve_moe_a2a_backend(
            "deepep",
            enable_expert_parallel=True,
            nnodes=1,
            data_parallel_size=1,
            data_parallel_size_local=1,
        )
        == "deepep"
    )


def test_require_single_node_raises_on_multi_node_hgx(monkeypatch):
    _pretend_platform(monkeypatch, cuda=True, arch="x86_64", capability=(10, 0))
    with pytest.raises(RuntimeError, match="one physical node"):
        require_single_node_without_mnnvl(2)


def test_require_single_node_allows_mnnvl_multi_node(monkeypatch):
    _pretend_platform(monkeypatch, cuda=True, arch="aarch64", capability=(10, 0))
    require_single_node_without_mnnvl(2)


def test_require_single_node_allows_unknown_node_count(monkeypatch):
    _pretend_platform(monkeypatch, cuda=True, arch="x86_64", capability=(10, 0))
    require_single_node_without_mnnvl(None)
