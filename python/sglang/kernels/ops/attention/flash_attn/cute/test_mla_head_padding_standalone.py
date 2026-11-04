"""CPU-only tests for the dense 2CTA MLA Q-head padding rule.

This module imports only the standard library and the sibling
``mla_head_padding.py`` by file path, so it can run in a minimal CI job
without CUDA/CuTe, torch, quack or the sglang package installed.
"""

import importlib.util
import pathlib
import sys

import pytest


cpu_only = pytest.mark.cpu_only


def _load_module():
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location(
        "_fa_cute_mla_head_padding_standalone", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mla_2cta_pads_dense_qheads(nheads_per_kv, num_heads_kv):
    return _load_module().mla_2cta_pads_dense_qheads(nheads_per_kv, num_heads_kv)


def mla_2cta_accepts_dense_qheads(nheads_per_kv, num_heads_kv):
    return _load_module().mla_2cta_accepts_dense_qheads(nheads_per_kv, num_heads_kv)


@cpu_only
@pytest.mark.parametrize(
    ("heads", "expected"),
    [
        (0, False),
        (-1, False),
        (1, False),
        (24, False),
        (64, False),
        (65, True),
        (72, True),
        (96, True),
        (127, True),
        (128, False),
        (129, False),
    ],
)
def test_pads_dense_qheads_boundaries(heads, expected):
    assert mla_2cta_pads_dense_qheads(heads, 1) is expected


@cpu_only
@pytest.mark.parametrize("heads", [65, 72, 96, 127])
def test_pads_dense_qheads_requires_mqa(heads):
    assert mla_2cta_pads_dense_qheads(heads, 1) is True
    assert mla_2cta_pads_dense_qheads(heads, 2) is False
    assert mla_2cta_pads_dense_qheads(heads, 8) is False


@cpu_only
def test_pads_dense_qheads_accepts_boolean_mqa_flag():
    assert mla_2cta_pads_dense_qheads(64, True) is False
    assert mla_2cta_pads_dense_qheads(65, True) is True
    assert mla_2cta_pads_dense_qheads(127, True) is True
    assert mla_2cta_pads_dense_qheads(128, True) is False
    assert mla_2cta_pads_dense_qheads(96, False) is False
    assert mla_2cta_pads_dense_qheads(0, True) is False
    assert mla_2cta_pads_dense_qheads(-1, True) is False


@cpu_only
@pytest.mark.parametrize(
    ("heads", "expected"),
    [
        (0, False),
        (-1, False),
        (1, True),
        (24, False),
        (64, True),
        (65, True),
        (96, True),
        (127, True),
        (128, True),
        (129, False),
    ],
)
def test_accepts_dense_qheads_boundaries(heads, expected):
    assert mla_2cta_accepts_dense_qheads(heads, 1) is expected


@cpu_only
def test_accepts_dense_qheads_requires_mqa_for_padding():
    assert mla_2cta_accepts_dense_qheads(96, 1) is True
    assert mla_2cta_accepts_dense_qheads(96, 2) is False
    assert mla_2cta_accepts_dense_qheads(24, 1) is False


@cpu_only
def test_predicate_resolves_without_config_module(monkeypatch):
    """A missing flash_attn.cute.config must not break the predicate."""
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", None)
    assert mla_2cta_pads_dense_qheads(65, 1) is True
    assert mla_2cta_pads_dense_qheads(64, 1) is False
    assert mla_2cta_pads_dense_qheads(128, 1) is False
    assert mla_2cta_pads_dense_qheads(96, 2) is False


@cpu_only
def test_predicate_resolves_without_sglang_or_cuda_deps(monkeypatch):
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "sglang", None)
    monkeypatch.setitem(sys.modules, "cutlass", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "quack", None)
    assert mla_2cta_pads_dense_qheads(65, True) is True
    assert mla_2cta_pads_dense_qheads(127, True) is True
    assert mla_2cta_pads_dense_qheads(128, True) is False
