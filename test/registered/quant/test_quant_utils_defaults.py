# SPDX-License-Identifier: Apache-2.0
"""Regression tests for mutable-default and shared-state fixes.

These tests are self-contained (stdlib + pytest only) and mirror the upstream
fixes that replaced mutable default arguments with ``None`` and stopped
mutating module-level dictionaries in place.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional

import pytest


# ---------------------------------------------------------------------------
# Mutable default arguments
# ---------------------------------------------------------------------------


def _collect(items: Optional[List[str]] = None) -> List[str]:
    """Mirror of the upstream pattern: default to None, allocate per call."""
    if items is None:
        items = []
    items.append("x")
    return items


def test_mutable_default_is_isolated_between_calls():
    first = _collect()
    second = _collect()
    assert first == ["x"]
    assert second == ["x"]
    assert first is not second


def test_mutable_default_respects_caller_list():
    caller = ["a"]
    result = _collect(caller)
    assert result is caller
    assert caller == ["a", "x"]


# ---------------------------------------------------------------------------
# Module-level dictionary must not be mutated in place
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
    source = (
        fused_mapping
        if fused_mapping is not None
        else _FALLBACK_FUSED_SHARDS
    )
    return {key: list(value) for key, value in source.items()}


def test_module_level_mapping_is_not_polluted():
    first = _copy_fused_mapping(None)
    first["qkv_proj"].append("extra")
    first["new_key"] = ["new"]

    second = _copy_fused_mapping(None)
    assert second["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
    assert "new_key" not in second
    assert "new_key" not in _FALLBACK_FUSED_SHARDS


def test_caller_mapping_is_not_mutated():
    caller: Dict[str, List[str]] = {"qkv_proj": ["q_proj"]}
    copied = _copy_fused_mapping(caller)
    copied["qkv_proj"].append("k_proj")
    assert caller == {"qkv_proj": ["q_proj"]}


# ---------------------------------------------------------------------------
# Dataclass optional fields: ``is not None`` instead of ``hasattr``
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402  (grouped with its tests)


@dataclass
class _LayerConfig:
    vocab_embedding: Optional[str] = None
    lm_head: Optional[str] = None


def _pad_embedding_lm_head(layer: _LayerConfig) -> bool:
    if layer.vocab_embedding is None or layer.lm_head is None:
        return False
    return True


@pytest.mark.parametrize(
    ("vocab_embedding", "lm_head", "expected"),
    [
        (None, None, False),
        ("embed", None, False),
        (None, "head", False),
        ("embed", "head", True),
    ],
)
def test_optional_dataclass_fields_use_is_not_none(
    vocab_embedding: Optional[str],
    lm_head: Optional[str],
    expected: bool,
) -> None:
    layer = _LayerConfig(vocab_embedding=vocab_embedding, lm_head=lm_head)
    assert _pad_embedding_lm_head(layer) is expected


def test_hasattr_would_have_misjudged_missing_field() -> None:
    layer = _LayerConfig()
    assert hasattr(layer, "vocab_embedding") is True
    assert layer.vocab_embedding is None
    assert _pad_embedding_lm_head(layer) is False
