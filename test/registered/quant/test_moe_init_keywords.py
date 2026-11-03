# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Mirrors the upstream fix where ``_process_single_layer`` passed stale
keyword names to ``_init_moe_module`` and raised ``TypeError``. These tests
are self-contained and do not import sglang.
"""

import pytest


def _init_moe_module(
    *,
    orig_router_weights,
    new_router_weights,
    orig_num_experts,
    new_num_experts,
):
    if orig_num_experts != len(orig_router_weights):
        raise ValueError("orig_num_experts does not match router weights")
    if new_num_experts != len(new_router_weights):
        raise ValueError("new_num_experts does not match router weights")
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


def test_stale_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weight=[1, 2, 3, 4],
            new_router_weight=[1, 2],
            orig_num_experts=4,
            new_num_experts=2,
        )


def test_mismatched_expert_count_raises_controlled_error():
    with pytest.raises(ValueError, match="orig_num_experts"):
        _init_moe_module(
            orig_router_weights=[1, 2],
            new_router_weights=[1, 2],
            orig_num_experts=4,
            new_num_experts=2,
        )
