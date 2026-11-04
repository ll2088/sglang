"""CPU-only boundary tests for the dense 2CTA MLA Q-head padding rule.

This module imports only the stdlib and the sibling ``mla_head_padding.py``
by file path, so it runs in a minimal CI job without CUDA/CuTe, torch, quack
or the sglang package. It is also loaded by ``test_pack_gqa.py`` to keep the
predicate boundary coverage in one place.
"""

import importlib.util
import pathlib

import pytest


cpu_only = pytest.mark.cpu_only


def _load_predicate_module():
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location(
        "_fa_cute_mla_head_padding_standalone", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_predicate_module()
mla_2cta_pads_dense_qheads = _MODULE.mla_2cta_pads_dense_qheads
mla_2cta_accepts_dense_qheads = _MODULE.mla_2cta_accepts_dense_qheads


@cpu_only
@pytest.mark.parametrize(
    ("heads", "heads_kv", "expected"),
    [
        (64, 1, False),
        (65, 1, True),
        (96, 1, True),
        (127, 1, True),
        (128, 1, False),
        (0, 1, False),
        (-1, 1, False),
        (96, 2, False),
        (96, 0, False),
    ],
)
def test_pads_dense_qheads_boundaries(heads, heads_kv, expected):
    assert mla_2cta_pads_dense_qheads(heads, heads_kv) is expected


@cpu_only
@pytest.mark.parametrize(
    ("heads", "mqa", "expected"),
    [
        (64, True, False),
        (65, True, True),
        (127, True, True),
        (128, True, False),
        (96, False, False),
        (0, True, False),
        (-1, True, False),
    ],
)
def test_pads_dense_qheads_accepts_boolean_mqa_flag(heads, mqa, expected):
    assert mla_2cta_pads_dense_qheads(heads, mqa) is expected


@cpu_only
@pytest.mark.parametrize(
    ("heads", "heads_kv", "expected"),
    [
        (128, 1, True),
        (64, 1, True),
        (96, 1, True),
        (65, 1, True),
        (127, 1, True),
        (24, 1, False),
        (96, 2, False),
        (0, 1, False),
        (-1, 1, False),
        (129, 1, False),
    ],
)
def test_accepts_dense_qheads_boundaries(heads, heads_kv, expected):
    assert mla_2cta_accepts_dense_qheads(heads, heads_kv) is expected


@cpu_only
def test_predicate_module_has_no_heavy_imports():
    """The rule module must stay importable without CUDA/CuTe/torch/quack."""
    source = pathlib.Path(__file__).with_name("mla_head_padding.py").read_text()
    for forbidden in ("import cutlass", "import torch", "import quack", "import sglang"):
        assert forbidden not in source
