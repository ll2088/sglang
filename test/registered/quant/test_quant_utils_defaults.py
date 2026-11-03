# SPDX-License-Identifier: Apache-2.0
"""Regression tests for mutable-default and module-level dict isolation.

These tests are intentionally self-contained (standard library + pytest only)
so they can run without importing sglang's heavy runtime dependencies.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest


# --- Pattern under test: mutable default arguments -------------------------


def _append_item(item: str, bucket: Optional[List[str]] = None) -> List[str]:
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket


def test_mutable_default_is_not_shared_between_calls():
    first = _append_item("a")
    second = _append_item("b")
    assert first == ["a"]
    assert second == ["b"]
    assert first is not second


def test_explicit_bucket_is_used_and_mutated():
    bucket: List[str] = []
    result = _append_item("x", bucket)
    assert result is bucket
    assert bucket == ["x"]


# --- Pattern under test: module-level dict mutated in place ---------------

_FALLBACK_MAPPING: Dict[str, List[str]] = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _copy_fused_mapping(
    fused_mapping: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    source = _FALLBACK_MAPPING if fused_mapping is None else fused_mapping
    return {key: list(value) for key, value in source.items()}


def test_module_level_mapping_is_not_mutated_by_callers():
    snapshot = {key: list(value) for key, value in _FALLBACK_MAPPING.items()}
    copied = _copy_fused_mapping()
    copied["qkv_proj"].append("extra")
    copied["new_key"] = ["new"]
    assert _FALLBACK_MAPPING == snapshot


def test_caller_mapping_is_not_mutated():
    caller = {"qkv_proj": ["q_proj"]}
    copied = _copy_fused_mapping(caller)
    copied["qkv_proj"].append("k_proj")
    assert caller == {"qkv_proj": ["q_proj"]}


@pytest.mark.parametrize("key", ["qkv_proj", "gate_up_proj"])
def test_copied_lists_are_independent(key):
    first = _copy_fused_mapping()
    second = _copy_fused_mapping()
    first[key].append("sentinel")
    assert "sentinel" not in second[key]
