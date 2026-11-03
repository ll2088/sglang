# SPDX-License-Identifier: Apache-2.0
"""Regression tests for fused_mapping isolation in quantization utils.

Both ``compressed_tensors.utils.should_ignore_layer`` and
``quark.utils.should_ignore_layer`` must copy caller-provided mappings so
that later mutation of the caller's dict cannot change the result of a call,
and so the call never mutates the caller's dict in place.
"""

import pytest

from sglang.srt.layers.quantization.compressed_tensors import utils as ct_utils
from sglang.srt.layers.quantization.quark import utils as quark_utils


@pytest.mark.parametrize(
    "should_ignore_layer",
    [ct_utils.should_ignore_layer, quark_utils.should_ignore_layer],
)
def test_should_ignore_layer_does_not_mutate_caller_mapping(should_ignore_layer):
    fused_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    snapshot = {key: list(value) for key, value in fused_mapping.items()}
    should_ignore_layer(
        "model.layers.0.self_attn.qkv_proj",
        ignore=[],
        fused_mapping=fused_mapping,
    )
    assert fused_mapping == snapshot


@pytest.mark.parametrize(
    "should_ignore_layer",
    [ct_utils.should_ignore_layer, quark_utils.should_ignore_layer],
)
def test_should_ignore_layer_rejects_non_mapping(should_ignore_layer):
    with pytest.raises(TypeError):
        should_ignore_layer(
            "model.layers.0.self_attn.qkv_proj",
            ignore=[],
            fused_mapping=["qkv_proj"],
        )


@pytest.mark.parametrize(
    "should_ignore_layer",
    [ct_utils.should_ignore_layer, quark_utils.should_ignore_layer],
)
def test_should_ignore_layer_uses_copied_mapping(should_ignore_layer):
    fused_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    result = should_ignore_layer(
        "model.layers.0.self_attn.qkv_proj",
        ignore=["model.layers.0.self_attn.q_proj"],
        fused_mapping=fused_mapping,
    )
    assert result is False
