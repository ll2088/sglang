# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Self-contained: mirrors the upstream fix where ``_process_single_layer``
passed stale keyword names to ``_init_moe_module`` and raised ``TypeError``.
The fixed contract accepts the current keyword names and raises a controlled
``TypeError`` for missing or invalid configuration.
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
    if orig_num_experts <= 0 or new_num_experts <= 0:
        raise TypeError("num_experts must be positive")
    if len(orig_router_weights) != orig_num_experts:
        raise TypeError("orig_router_weights length mismatch")
    if len(new_router_weights) != new_num_experts:
        raise TypeError("new_router_weights length mismatch")
    return {
        "orig_router_weights": orig_router_weights,
        "new_router_weights": new_router_weights,
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def test_current_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[0, 1, 2, 3],
        new_router_weights=[0, 1],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2


def test_legacy_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weight=[0, 1],
            new_router_weight=[0],
            orig_num_experts=2,
            new_num_experts=1,
        )


@pytest.mark.parametrize(
    ("orig_num_experts", "new_num_experts"),
    [(None, 1), (1, None), (0, 1), (1, 0)],
)
def test_invalid_expert_counts_raise_controlled_type_error(
    orig_num_experts, new_num_experts
):
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weights=[],
            new_router_weights=[],
            orig_num_experts=orig_num_experts,
            new_num_experts=new_num_experts,
        )


def test_router_weight_length_mismatch_raises_type_error():
    with pytest.raises(TypeError, match="length mismatch"):
        _init_moe_module(
            orig_router_weights=[0],
            new_router_weights=[0],
            orig_num_experts=2,
            new_num_experts=1,
        )
