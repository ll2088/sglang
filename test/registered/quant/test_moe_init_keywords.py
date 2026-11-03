# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Mirrors the upstream fix where ``_init_moe_module`` was called with stale
keyword names, raising ``TypeError``. The helper below accepts the current
keyword set and raises a controlled ``TypeError`` for missing config.
"""

import pytest


def _init_moe_module(
    *,
    orig_router_weights,
    new_router_weights,
    orig_num_experts,
    new_num_experts,
):
    if orig_num_experts is None or new_num_experts is None:
        raise TypeError("num_experts must be provided")
    if orig_num_experts < new_num_experts:
        raise ValueError("cannot grow expert count")
    return {
        "orig_router_weights": orig_router_weights,
        "new_router_weights": new_router_weights,
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def test_current_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[1, 2, 3, 4],
        new_router_weights=[1, 2],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2


def test_legacy_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weight=[1, 2],
            new_router_weight=[1],
            orig_num_experts=2,
            new_num_experts=1,
        )


def test_missing_expert_count_raises_controlled_type_error():
    with pytest.raises(TypeError, match="num_experts"):
        _init_moe_module(
            orig_router_weights=[],
            new_router_weights=[],
            orig_num_experts=None,
            new_num_experts=1,
        )
