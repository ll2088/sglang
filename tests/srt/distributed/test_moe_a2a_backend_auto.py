# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

"""Unit tests for the MoE all2all backend auto-selection policy."""

import pytest

from sglang.srt.distributed.device_communicators import all2all_auto


@pytest.fixture(autouse=True)
def _clear_probes():
    all2all_auto.clear_platform_probes()
    yield
    all2all_auto.clear_platform_probes()


def _pretend(monkeypatch, *, cuda, arch, capability, one_sided=True):
    monkeypatch.setattr(all2all_auto, "_is_cuda", lambda: cuda)
    monkeypatch.setattr(all2all_auto, "_cpu_arch", lambda: arch)
    monkeypatch.setattr(all2all_auto, "_device_capability", lambda: capability)
    monkeypatch.setattr(all2all_auto, "_has_one_sided_support", lambda: one_sided)
    all2all_auto.clear_platform_probes()


def _resolve(**overrides):
    kwargs = dict(
        enable_expert_parallel=True,
        enable_elastic_ep=False,
        nnodes=1,
        data_parallel_size=1,
        data_parallel_size_local=1,
        distributed_executor_backend="mp",
    )
    kwargs.update(overrides)
    return all2all_auto.resolve_all2all_backend("auto", **kwargs)


@pytest.mark.parametrize(
    ("arch", "capability"),
    [
        ("aarch64", (10, 0)),  # GB200
        ("aarch64", (10, 3)),  # GB300
        ("aarch64", (10, 7)),  # VR200
        ("x86_64", (10, 0)),  # HGX B200
        ("x86_64", (10, 3)),  # HGX B300
    ],
)
def test_auto_enables_one_sided_on_blackwell_rubin(monkeypatch, arch, capability):
    _pretend(monkeypatch, cuda=True, arch=arch, capability=capability)
    backend, auto = _resolve()
    assert backend == all2all_auto.ONE_SIDED_BACKEND
    assert auto is True


@pytest.mark.parametrize(
    ("arch", "capability"),
    [
        ("aarch64", (12, 1)),  # GB10
        ("aarch64", (9, 0)),  # GH200
        ("x86_64", (9, 0)),  # Hopper
        ("x86_64", (8, 0)),  # Ampere
    ],
)
def test_auto_falls_back_on_other_platforms(monkeypatch, arch, capability):
    _pretend(monkeypatch, cuda=True, arch=arch, capability=capability)
    backend, auto = _resolve()
    assert backend == all2all_auto.DEFAULT_BACKEND
    assert auto is False


def test_auto_falls_back_without_cuda(monkeypatch):
    _pretend(monkeypatch, cuda=False, arch="x86_64", capability=None)
    backend, auto = _resolve()
    assert backend == all2all_auto.DEFAULT_BACKEND
    assert auto is False


def test_auto_falls_back_without_flashinfer(monkeypatch):
    _pretend(monkeypatch, cuda=True, arch="aarch64", capability=(10, 0), one_sided=False)
    backend, auto = _resolve()
    assert backend == all2all_auto.DEFAULT_BACKEND
    assert auto is False


def test_auto_falls_back_for_elastic_ep(monkeypatch):
    _pretend(monkeypatch, cuda=True, arch="aarch64", capability=(10, 0))
    backend, auto = _resolve(enable_elastic_ep=True)
    assert backend == all2all_auto.DEFAULT_BACKEND
    assert auto is False


def test_auto_falls_back_without_expert_parallel(monkeypatch):
    _pretend(monkeypatch, cuda=True, arch="aarch64", capability=(10, 0))
    backend, auto = _resolve(enable_expert_parallel=False)
    assert backend == all2all_auto.DEFAULT_BACKEND
    assert auto is False


def test_auto_falls_back_on_multi_node_hgx(monkeypatch):
    _pretend(monkeypatch, cuda=True, arch="x86_64", capability=(10, 0))
    backend, auto = _resolve(nnodes=2)
    assert backend == all2all_auto.DEFAULT_BACKEND
    assert auto is False


def test_auto_allows_multi_node_on_mnnvl(monkeypatch):
    _pretend(monkeypatch, cuda=True, arch="aarch64", capability=(10, 0))
    backend, auto = _resolve(nnodes=2)
    assert backend == all2all_auto.ONE_SIDED_BACKEND
    assert auto is True


def test_explicit_backend_overrides_auto(monkeypatch):
    _pretend(monkeypatch, cuda=True, arch="aarch64", capability=(10, 0))
    backend, auto = all2all_auto.resolve_all2all_backend(
        "deepep",
        enable_expert_parallel=True,
        enable_elastic_ep=False,
        nnodes=1,
        data_parallel_size=1,
        data_parallel_size_local=1,
        distributed_executor_backend="mp",
    )
    assert backend == "deepep"
    assert auto is False


def test_single_node_guard_raises_on_multi_node_hgx(monkeypatch):
    _pretend(monkeypatch, cuda=True, arch="x86_64", capability=(10, 0))
    with pytest.raises(RuntimeError, match="one physical node"):
        all2all_auto.require_single_node_without_mnnvl(node_count=2, auto=True)


def test_single_node_guard_allows_mnnvl(monkeypatch):
    _pretend(monkeypatch, cuda=True, arch="aarch64", capability=(10, 0))
    all2all_auto.require_single_node_without_mnnvl(node_count=4, auto=True)


def test_single_node_guard_allows_single_node(monkeypatch):
    _pretend(monkeypatch, cuda=True, arch="x86_64", capability=(10, 0))
    all2all_auto.require_single_node_without_mnnvl(node_count=1, auto=True)
