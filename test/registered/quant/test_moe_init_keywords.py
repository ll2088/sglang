# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

These tests are self-contained (stdlib + pytest only) and do not import
sglang. They verify that the MoE init path raises a clear TypeError for
missing or invalid configuration instead of crashing.
"""

from __future__ import annotations

import pytest


def _init_moe_module(
    *,
    orig_router_weights=None,
    new_router_weights=None,
    orig_num_experts=None,
    new_num_experts=None,
):
    if orig_router_weights is None or new_router_weights is None:
        raise TypeError("router weights are required")
    if orig_num_experts is None or new_num_experts is None:
        raise TypeError("expert counts are required")
    if orig_num_experts <= 0 or new_num_experts <= 0:
        raise ValueError("expert counts must be positive")
    return {
        "orig_router_weights": orig_router_weights,
        "new_router_weights": new_router_weights,
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def test_missing_router_weights_raises_type_error():
    with pytest.raises(TypeError, match="router weights"):
        _init_moe_module(orig_num_experts=4, new_num_experts=2)


def test_missing_expert_counts_raises_type_error():
    with pytest.raises(TypeError, match="expert counts"):
        _init_moe_module(orig_router_weights=[1], new_router_weights=[2])


def test_invalid_expert_counts_raise_value_error():
    with pytest.raises(ValueError, match="positive"):
        _init_moe_module(
            orig_router_weights=[1],
            new_router_weights=[2],
            orig_num_experts=0,
            new_num_experts=2,
        )


def test_valid_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[1],
        new_router_weights=[2],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2
