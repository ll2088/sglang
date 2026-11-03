# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Mirrors the upstream fix pattern: MoE initialization must accept the current
keyword names and raise a clear ``TypeError`` for stale keyword names instead
of crashing deep inside the call.
"""

import pytest


def _init_moe_module(
    *,
    orig_router_weights,
    new_router_weights,
    orig_num_experts,
    new_num_experts,
):
    if orig_num_experts <= 0 or new_num_experts <= 0:
        raise ValueError("num_experts must be positive")
    return {
        "orig_router_weights": orig_router_weights,
        "new_router_weights": new_router_weights,
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def test_current_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[1, 2, 3, 4],
        new_router_weights=[1, 3],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2


def test_stale_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weight=[1, 2, 3, 4],
            new_router_weight=[1, 3],
            orig_num_experts=4,
            new_num_experts=2,
        )


def test_invalid_expert_count_raises_value_error():
    with pytest.raises(ValueError, match="num_experts must be positive"):
        _init_moe_module(
            orig_router_weights=[],
            new_router_weights=[],
            orig_num_experts=0,
            new_num_experts=0,
        )
