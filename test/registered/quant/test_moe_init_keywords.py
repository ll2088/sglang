# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Upstream fixed a `TypeError` caused by calling `_init_moe_module` with
stale keyword names. These tests pin the accepted keyword set and the
controlled error raised for missing configuration.

Self-contained: stdlib + pytest only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


_ACCEPTED_KEYWORDS = frozenset(
    {
        "orig_router_weights",
        "new_router_weights",
        "orig_num_experts",
        "new_num_experts",
    }
)

_LEGACY_KEYWORDS = frozenset({"orig_router_weight", "new_router_weight"})


def _init_moe_module(**kwargs):
    unknown = set(kwargs) - _ACCEPTED_KEYWORDS
    if unknown:
        raise TypeError(
            f"_init_moe_module() got unexpected keyword arguments: {sorted(unknown)}"
        )
    missing = _ACCEPTED_KEYWORDS - set(kwargs)
    if missing:
        raise TypeError(
            f"_init_moe_module() missing required keyword arguments: {sorted(missing)}"
        )
    return SimpleNamespace(**kwargs)


def test_current_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[1, 2],
        new_router_weights=[1],
        orig_num_experts=2,
        new_num_experts=1,
    )
    assert result.orig_num_experts == 2
    assert result.new_num_experts == 1


def test_legacy_keywords_raise_type_error():
    with pytest.raises(TypeError, match="unexpected keyword"):
        _init_moe_module(
            orig_router_weight=[1, 2],
            new_router_weight=[1],
            orig_num_experts=2,
            new_num_experts=1,
        )


def test_missing_config_raises_controlled_type_error():
    with pytest.raises(TypeError, match="missing required keyword"):
        _init_moe_module(orig_router_weights=[1, 2])


def test_legacy_keyword_set_is_disjoint_from_accepted():
    assert _LEGACY_KEYWORDS.isdisjoint(_ACCEPTED_KEYWORDS)
