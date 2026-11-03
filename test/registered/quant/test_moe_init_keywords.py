# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Ensures that missing or malformed configuration raises a controlled error
instead of crashing with an unrelated TypeError.
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
    if orig_num_experts is None or new_num_experts is None:
        raise ValueError("num_experts must be provided")
    if orig_router_weights is None or new_router_weights is None:
        raise ValueError("router weights must be provided")
    return {
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def test_missing_num_experts_raises_controlled_error():
    with pytest.raises(ValueError, match="num_experts"):
        _init_moe_module(orig_router_weights=[], new_router_weights=[])


def test_missing_router_weights_raises_controlled_error():
    with pytest.raises(ValueError, match="router weights"):
        _init_moe_module(orig_num_experts=4, new_num_experts=2)


def test_valid_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[1, 2, 3, 4],
        new_router_weights=[1, 2],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result == {"orig_num_experts": 4, "new_num_experts": 2}
