# SPDX-License-Identifier: Apache-2.0
"""Regression tests for mutable-default and shared-state fixes.

These tests are intentionally self-contained (pytest + stdlib only) so they
can be collected without importing the full sglang runtime. They mirror the
upstream fix pattern: mutable defaults become ``None`` and module-level
mappings are copied before use.
"""

from types import MappingProxyType

import pytest


_FALLBACK_FUSED_SHARDS = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _copy_fused_mapping(fused_mapping=None):
    source = (
        fused_mapping
        if fused_mapping is not None
        else MappingProxyType(_FALLBACK_FUSED_SHARDS)
    )
    return {key: list(value) for key, value in source.items()}


def _append_default(value, items=None):
    if items is None:
        items = []
    items.append(value)
    return items


def test_mutable_default_is_isolated_between_calls():
    first = _append_default(1)
    second = _append_default(2)
    assert first == [1]
    assert second == [2]
    assert first is not second


def test_fallback_mapping_is_not_mutated_in_place():
    before = {key: list(value) for key, value in _FALLBACK_FUSED_SHARDS.items()}
    copied = _copy_fused_mapping()
    copied["qkv_proj"].append("extra_proj")
    copied["new_proj"] = ["x"]
    assert _FALLBACK_FUSED_SHARDS == before
    assert "new_proj" not in _FALLBACK_FUSED_SHARDS


def test_caller_mapping_is_copied():
    caller = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    copied = _copy_fused_mapping(caller)
    copied["qkv_proj"].append("extra_proj")
    assert caller == {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}


@pytest.mark.parametrize("key", sorted(_FALLBACK_FUSED_SHARDS))
def test_each_fallback_key_is_independent(key):
    first = _copy_fused_mapping()
    second = _copy_fused_mapping()
    first[key].append("sentinel")
    assert "sentinel" not in second[key]
