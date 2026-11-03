# SPDX-License-Identifier: Apache-2.0
"""Regression tests for mutable-default and module-level state fixes."""

from types import MappingProxyType

import pytest

pytest.importorskip("compressed_tensors")

from sglang.srt.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
    should_ignore_layer as ct_should_ignore_layer,
)
from sglang.srt.layers.quantization.quark.utils import (
    should_ignore_layer as quark_should_ignore_layer,
)
from sglang.srt.layers.quantization.utils import (
    _FALLBACK_FUSED_SHARDS,
    is_layer_skipped,
)


class _DummyModule:
    __class__ = type("Dummy", (), {"__name__": "Dummy"})


def test_is_layer_skipped_does_not_mutate_fallback_mapping():
    before = {k: list(v) for k, v in _FALLBACK_FUSED_SHARDS.items()}
    assert is_layer_skipped("model.layers.0.mlp.gate_up_proj", ["mlp.gate"])
    assert {k: list(v) for k, v in _FALLBACK_FUSED_SHARDS.items()} == before
    # A second call must observe the same pristine module-level mapping.
    assert is_layer_skipped("model.layers.0.mlp.gate_up_proj", ["mlp.gate"])
    assert {k: list(v) for k, v in _FALLBACK_FUSED_SHARDS.items()} == before
    # The fallback mapping must not be mutated even when a caller passes a
    # mapping that happens to be the module-level object itself.
    assert is_layer_skipped(
        "model.layers.0.mlp.gate_up_proj", ["mlp.gate"], _FALLBACK_FUSED_SHARDS
    )
    assert {k: list(v) for k, v in _FALLBACK_FUSED_SHARDS.items()} == before


def test_is_layer_skipped_accepts_none_fused_mapping():
    assert is_layer_skipped("model.layers.0.self_attn.q_proj", ["q_proj"], None)
    assert not is_layer_skipped("model.layers.0.self_attn.q_proj", [], None)


def test_compressed_tensors_should_ignore_layer_defaults_are_isolated():
    assert not ct_should_ignore_layer("model.layers.0.mlp.down_proj")
    assert ct_should_ignore_layer("model.layers.0.mlp.down_proj", ["down_proj"])
    # A second call without arguments must not see the previous ignore list.
    assert not ct_should_ignore_layer("model.layers.0.mlp.down_proj")
    # Passing an explicit empty list must also not leak into later calls.
    assert not ct_should_ignore_layer("model.layers.0.mlp.down_proj", [])
    assert not ct_should_ignore_layer("model.layers.0.mlp.down_proj")


def test_compressed_tensors_should_ignore_layer_does_not_mutate_fused_mapping():
    fused = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    before = {k: list(v) for k, v in fused.items()}
    assert ct_should_ignore_layer(
        "model.layers.0.self_attn.qkv_proj", ["q_proj", "k_proj", "v_proj"], fused
    )
    assert {k: list(v) for k, v in fused.items()} == before


def test_compressed_tensors_find_matched_target_accepts_none_mapping():
    target = find_matched_target(
        "model.layers.0.mlp.down_proj", _DummyModule(), ["down_proj"], None
    )
    assert target == "down_proj"


def test_quark_should_ignore_layer_defaults_are_isolated():
    assert not quark_should_ignore_layer("model.layers.0.mlp.down_proj")
    assert quark_should_ignore_layer("model.layers.0.mlp.down_proj", ["down_proj"])
    assert not quark_should_ignore_layer("model.layers.0.mlp.down_proj")
    assert not quark_should_ignore_layer("model.layers.0.mlp.down_proj", [])
    assert not quark_should_ignore_layer("model.layers.0.mlp.down_proj")


def test_quark_should_ignore_layer_accepts_none_fused_mapping():
    assert quark_should_ignore_layer(
        "model.layers.0.self_attn.q_proj", ["q_proj"], None
    )


def test_quark_should_ignore_layer_does_not_mutate_fused_mapping():
    fused = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    before = {k: list(v) for k, v in fused.items()}
    assert quark_should_ignore_layer(
        "model.layers.0.self_attn.qkv_proj", ["q_proj", "k_proj", "v_proj"], fused
    )
    assert {k: list(v) for k, v in fused.items()} == before


@pytest.mark.parametrize("mapping", [None, MappingProxyType({})])
def test_compressed_tensors_should_ignore_layer_accepts_empty_mapping(mapping):
    assert not ct_should_ignore_layer("model.layers.0.mlp.down_proj", [], mapping)
