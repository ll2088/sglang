# SPDX-License-Identifier: Apache-2.0
"""Regression tests for mutable defaults and module-level dict isolation.

These tests are self-contained (stdlib + pytest only) and do not import
sglang, so they can run in the minimal CI environment.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

_FALLBACK_FUSED_SHARDS = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _copy_fused_mapping(fused_mapping=None):
    if fused_mapping is not None and not isinstance(fused_mapping, dict):
        raise TypeError("fused_mapping must be a mapping")
    source = (
        fused_mapping
        if fused_mapping is not None
        else MappingProxyType(_FALLBACK_FUSED_SHARDS)
    )
    return {key: list(value) for key, value in source.items()}


def _with_mutable_default(value, extra=None):
    if extra is None:
        extra = {}
    extra["value"] = value
    return extra


def test_mutable_default_is_isolated_between_calls():
    first = _with_mutable_default(1)
    second = _with_mutable_default(2)
    assert first == {"value": 1}
    assert second == {"value": 2}
    assert first is not second


def test_module_level_fallback_is_not_mutated():
    original = {key: list(value) for key, value in _FALLBACK_FUSED_SHARDS.items()}
    copied = _copy_fused_mapping()
    copied["qkv_proj"].append("extra")
    assert _FALLBACK_FUSED_SHARDS == original


def test_caller_mapping_is_copied():
    caller = {"qkv_proj": ["q_proj"]}
    copied = _copy_fused_mapping(caller)
    copied["qkv_proj"].append("k_proj")
    assert caller == {"qkv_proj": ["q_proj"]}


def test_invalid_mapping_raises_type_error():
    with pytest.raises(TypeError):
        _copy_fused_mapping(["not", "a", "mapping"])
