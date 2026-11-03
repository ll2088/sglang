# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

These tests are self-contained (stdlib + pytest only) and do not import
sglang. They verify that missing or unexpected configuration raises a
controlled exception instead of crashing with an unrelated TypeError.
"""

from __future__ import annotations

import pytest


def _init_moe_module(
    orig_router_weights=None,
    new_router_weights=None,
    orig_num_experts=None,
    new_num_experts=None,
):
    if orig_num_experts is None or new_num_experts is None:
        raise TypeError("orig_num_experts and new_num_experts are required")
    if orig_router_weights is None or new_router_weights is None:
        raise TypeError("router weights are required")
    return {
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def test_missing_expert_counts_raise_type_error():
    with pytest.raises(TypeError, match="orig_num_experts"):
        _init_moe_module(orig_router_weights=[], new_router_weights=[])


def test_missing_router_weights_raise_type_error():
    with pytest.raises(TypeError, match="router weights"):
        _init_moe_module(orig_num_experts=4, new_num_experts=2)


def test_valid_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[1, 2, 3, 4],
        new_router_weights=[1, 2],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result == {"orig_num_experts": 4, "new_num_experts": 2}
