# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Mirrors the upstream fix where `_process_single_layer` passed stale keyword
names to `_init_moe_module`, raising `TypeError`. These tests are
self-contained (stdlib + pytest only) and do not import sglang.
"""

from __future__ import annotations

import pytest


_CURRENT_KEYWORDS = (
    "orig_router_weights",
    "new_router_weights",
    "orig_num_experts",
    "new_num_experts",
)

_LEGACY_KEYWORDS = ("orig_router_weight", "new_router_weight")


def _init_moe_module(**kwargs):
    missing = [k for k in _CURRENT_KEYWORDS if k not in kwargs]
    if missing:
        raise TypeError(f"missing required keyword(s): {missing}")
    return {k: kwargs[k] for k in _CURRENT_KEYWORDS}


def _process_single_layer(use_legacy_keywords):
    if use_legacy_keywords:
        return _init_moe_module(
            orig_router_weight=None,
            new_router_weight=None,
            orig_num_experts=4,
            new_num_experts=2,
        )
    return _init_moe_module(
        orig_router_weights=None,
        new_router_weights=None,
        orig_num_experts=4,
        new_num_experts=2,
    )


def test_legacy_keywords_raise_type_error():
    with pytest.raises(TypeError, match="missing required keyword"):
        _process_single_layer(use_legacy_keywords=True)


def test_current_keywords_succeed():
    result = _process_single_layer(use_legacy_keywords=False)
    assert set(result) == set(_CURRENT_KEYWORDS)
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2


def test_legacy_keyword_names_are_not_accepted():
    for legacy in _LEGACY_KEYWORDS:
        assert legacy not in _CURRENT_KEYWORDS
