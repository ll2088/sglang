import importlib.util
import pathlib

import pytest


def _load_predicate():
    """Load the predicate without importing the sglang package (no CUDA deps)."""
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location("fa_cute_mla_head_padding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mla_2cta_pads_dense_qheads


mla_2cta_pads_dense_qheads = _load_predicate()


@pytest.mark.parametrize(
    ("nheads_per_kv", "num_heads_kv", "expected"),
    [
        (64, 1, False),
        (65, 1, True),
        (96, 1, True),
        (127, 1, True),
        (128, 1, False),
        (24, 1, False),
        (0, 1, False),
        (-1, 1, False),
        (129, 1, False),
        (96, 2, False),
        (65, 2, False),
        (96, 0, False),
        (64, True, False),
        (65, True, True),
        (96, True, True),
        (127, True, True),
        (128, True, False),
        (0, True, False),
        (-1, True, False),
        (96, False, False),
    ],
)
def test_mla_2cta_pads_dense_qheads(nheads_per_kv, num_heads_kv, expected):
    assert mla_2cta_pads_dense_qheads(nheads_per_kv, num_heads_kv) is expected


def test_mla_2cta_pads_dense_qheads_accepts_boolean_mqa_flag():
    # The usability check queries the predicate with a boolean MQA flag.
    assert mla_2cta_pads_dense_qheads(64, True) is False
    assert mla_2cta_pads_dense_qheads(65, True) is True
    assert mla_2cta_pads_dense_qheads(127, True) is True
    assert mla_2cta_pads_dense_qheads(128, True) is False
    assert mla_2cta_pads_dense_qheads(96, False) is False
    assert mla_2cta_pads_dense_qheads(0, True) is False
    assert mla_2cta_pads_dense_qheads(-1, True) is False
