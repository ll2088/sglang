# SPDX-License-Identifier: Apache-2.0
"""Regression tests for mutable-default and module-level shared-state bugs.

These tests are self-contained (stdlib + pytest only) so they can run without
installing the full sglang dependency tree. They mirror the upstream fixes:

* mutable default arguments (``dict``/``list``/``set``) must be ``None`` and
  materialized per call;
* module-level mapping defaults must be copied before being handed out, so a
  caller mutating the returned view cannot poison later calls.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Dict, List, Mapping, Optional

import pytest


# ---------------------------------------------------------------------------
# Mutable default arguments
# ---------------------------------------------------------------------------


def _collect(items: Optional[List[str]] = None) -> List[str]:
    """Mirror of the upstream pattern: ``None`` default, fresh list per call."""
    if items is None:
        items = []
    items.append("x")
    return items


def _merge_kwargs(kwargs: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    if kwargs is None:
        kwargs = {}
    kwargs["seen"] = kwargs.get("seen", 0) + 1
    return kwargs


def test_list_default_is_not_shared_between_calls():
    first = _collect()
    second = _collect()
    assert first == ["x"]
    assert second == ["x"]
    assert first is not second


def test_list_default_is_not_shared_with_caller_argument():
    caller = ["a"]
    result = _collect(caller)
    assert result is caller
    assert caller == ["a", "x"]
    # A later call without an argument must not observe the caller's list.
    assert _collect() == ["x"]


def test_dict_default_is_not_shared_between_calls():
    first = _merge_kwargs()
    second = _merge_kwargs()
    assert first == {"seen": 1}
    assert second == {"seen": 1}
    assert first is not second


def test_dict_default_does_not_leak_into_next_call():
    first = _merge_kwargs()
    first["custom"] = True
    second = _merge_kwargs()
    assert "custom" not in second


# ---------------------------------------------------------------------------
# Module-level mapping must be copied, never mutated in place
# ---------------------------------------------------------------------------

_FALLBACK_FUSED_SHARDS: Mapping[str, List[str]] = MappingProxyType(
    {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
)


def _copy_fused_mapping(
    fused_mapping: Optional[Mapping[str, List[str]]],
) -> Dict[str, List[str]]:
    if fused_mapping is not None and not isinstance(fused_mapping, Mapping):
        raise TypeError(
            "fused_mapping must be a mapping of fused layer names to shard "
            f"names, got {type(fused_mapping).__name__}"
        )
    source = (
        fused_mapping
        if fused_mapping is not None
        else _FALLBACK_FUSED_SHARDS
    )
    return {key: list(value) for key, value in source.items()}


def test_module_level_mapping_is_not_mutated_by_callers():
    snapshot = {key: list(value) for key, value in _FALLBACK_FUSED_SHARDS.items()}
    view = _copy_fused_mapping(None)
    view["qkv_proj"].append("extra")
    view["new_key"] = ["new"]
    assert {key: list(value) for key, value in _FALLBACK_FUSED_SHARDS.items()} == snapshot


def test_module_level_mapping_is_isolated_between_calls():
    first = _copy_fused_mapping(None)
    first["qkv_proj"].append("extra")
    second = _copy_fused_mapping(None)
    assert second["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
    assert "extra" not in second["qkv_proj"]


def test_caller_mapping_is_copied_not_aliased():
    caller = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    view = _copy_fused_mapping(caller)
    view["qkv_proj"].append("extra")
    assert caller["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]


def test_non_mapping_argument_is_rejected():
    with pytest.raises(TypeError, match="fused_mapping must be a mapping"):
        _copy_fused_mapping(["not", "a", "mapping"])  # type: ignore[arg-type]
