# SPDX-License-Identifier: Apache-2.0
"""Regression tests for mutable defaults and module-level dict isolation.

These tests are intentionally self-contained: they do not import ``sglang`` so
that they can run without the full runtime dependency tree. The patterns below
mirror the production helpers in
``sglang.srt.layers.quantization.utils`` (``_FALLBACK_FUSED_SHARDS`` and
``is_layer_skipped``).
"""

from types import MappingProxyType

import pytest


_FALLBACK_FUSED_SHARDS = MappingProxyType(
    {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
)


def _effective_fused(fused_mapping=None, proj_name="qkv_proj"):
    if fused_mapping is not None and proj_name in fused_mapping:
        return dict(fused_mapping)
    return dict(_FALLBACK_FUSED_SHARDS)


def _append_default(value, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(value)
    return bucket


def test_mutable_default_is_isolated_between_calls():
    first = _append_default(1)
    second = _append_default(2)
    assert first == [1]
    assert second == [2]
    assert first is not second


def test_module_level_mapping_is_not_mutated_in_place():
    before = dict(_FALLBACK_FUSED_SHARDS)
    result = _effective_fused()
    result["qkv_proj"].append("extra")
    assert dict(_FALLBACK_FUSED_SHARDS) == before
    assert _FALLBACK_FUSED_SHARDS["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]


def test_caller_mapping_is_copied():
    caller = {"qkv_proj": ["q_proj"]}
    result = _effective_fused(caller)
    result["qkv_proj"].append("k_proj")
    assert caller == {"qkv_proj": ["q_proj"]}


@pytest.mark.parametrize("proj_name", ["qkv_proj", "gate_up_proj"])
def test_fallback_mapping_covers_fused_layers(proj_name):
    assert proj_name in _FALLBACK_FUSED_SHARDS
