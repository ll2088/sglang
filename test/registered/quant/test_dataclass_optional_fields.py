# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field checks."""

from types import SimpleNamespace

from sglang.srt.models.inkling_common.quantization.config import (
    InklingQuantizationConfigBase,
)


class _Config(InklingQuantizationConfigBase):
    def __init__(self, exclude_modules=None):
        if exclude_modules is not None:
            self.exclude_modules = exclude_modules


def test_exclude_layer_without_exclude_modules_attribute():
    config = _Config()
    assert config.exclude_layer("model.layers.0.mlp.down_proj") is False


def test_exclude_layer_with_empty_exclude_modules():
    config = _Config(exclude_modules=[])
    assert config.exclude_layer("model.layers.0.mlp.down_proj") is False


def test_exclude_layer_matches_module():
    config = _Config(exclude_modules=["mlp.down_proj"])
    assert config.exclude_layer("model.layers.0.mlp.down_proj") is True
    assert config.exclude_layer("model.layers.0.mlp.gate_up_proj") is False


def test_exclude_layer_matches_language_model_prefix():
    config = _Config(exclude_modules=["mlp.down_proj"])
    assert config.exclude_layer("language_model.model.layers.0.mlp.down_proj") is True
