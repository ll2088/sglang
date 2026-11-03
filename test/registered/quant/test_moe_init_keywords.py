# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Mirrors the upstream Puzzletron fix: callers must pass the current
``orig_*``/``new_*`` keyword names, and a stale keyword set must raise a
clear ``TypeError`` instead of crashing deeper in the call stack.
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
        orig_router_weights=[0, 1],
        new_router_weights=[0],
        orig_num_experts=2,
        new_num_experts=1,
    )
    assert result["orig_num_experts"] == 2
    assert result["new_num_experts"] == 1


def test_stale_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weight=[0, 1],
            new_router_weight=[0],
            orig_num_experts=2,
            new_num_experts=1,
        )


@pytest.mark.parametrize(
    ("orig_num_experts", "new_num_experts"),
    [(0, 1), (1, 0), (-1, 1)],
)
def test_invalid_expert_counts_raise_value_error(orig_num_experts, new_num_experts):
    with pytest.raises(ValueError, match="num_experts"):
        _init_moe_module(
            orig_router_weights=[],
            new_router_weights=[],
            orig_num_experts=orig_num_experts,
            new_num_experts=new_num_experts,
        )
