# SPDX-License-Identifier: Apache-2.0
"""Regression tests for mutable-default and module-level mapping fixes.

These tests are self-contained (stdlib + pytest only) and mirror the upstream
fix that replaced mutable defaults with ``None`` and copied module-level
mappings before use.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

_FALLBACK_FUSED_SHARDS = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _copy_fused_mapping(fused_mapping=None):
    if fused_mapping is not None and not isinstance(fused_mapping, Mapping):
        raise TypeError(
            "fused_mapping must be a mapping of fused layer names to shard "
            f"names, got {type(fused_mapping).__name__}"
        )
    source = (
        fused_mapping
        if fused_mapping is not None
        else MappingProxyType(_FALLBACK_FUSED_SHARDS)
    )
    return {key: list(value) for key, value in source.items()}


def _make_mapping(func_kwargs=None):
    return {} if func_kwargs is None else func_kwargs


def test_mutable_default_is_isolated_between_calls():
    first = _make_mapping()
    first["custom_option"] = True
    second = _make_mapping()
    assert second == {}


def test_module_level_mapping_is_not_mutated_in_place():
    original = {key: list(value) for key, value in _FALLBACK_FUSED_SHARDS.items()}
    copied = _copy_fused_mapping()
    copied["qkv_proj"].append("extra_proj")
    assert _FALLBACK_FUSED_SHARDS == original


def test_caller_mapping_is_copied():
    caller = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    copied = _copy_fused_mapping(caller)
    copied["qkv_proj"].append("extra_proj")
    assert caller["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]


@pytest.mark.parametrize("value", [None, {}, {"qkv_proj": ["q_proj"]}])
def test_copy_fused_mapping_accepts_optional_inputs(value):
    result = _copy_fused_mapping(value)
    assert isinstance(result, dict)
    for key, shards in result.items():
        assert isinstance(key, str)
        assert isinstance(shards, list)
