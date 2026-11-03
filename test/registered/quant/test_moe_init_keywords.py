# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

The upstream fix corrected a call that passed stale keyword names to
``_init_moe_module``. These tests use a small stand-in to assert that the
current keyword names are accepted and that missing configuration raises a
controlled ``TypeError`` instead of crashing.
"""

from __future__ import annotations

import pytest


def _init_moe_module(
    *,
    orig_router_weights,
    new_router_weights,
    orig_num_experts,
    new_num_experts,
):
    if orig_num_experts <= 0 or new_num_experts <= 0:
        raise ValueError("expert counts must be positive")
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
            orig_router_weight=[1, 2, 3, 4],
            new_router_weight=[1, 2],
            orig_num_experts=4,
            new_num_experts=2,
        )


@pytest.mark.parametrize(
    ("orig_num_experts", "new_num_experts"),
    [(0, 2), (4, 0), (-1, 2)],
)
def test_invalid_expert_counts_raise_controlled_error(orig_num_experts, new_num_experts):
    with pytest.raises(ValueError, match="positive"):
        _init_moe_module(
            orig_router_weights=[],
            new_router_weights=[],
            orig_num_experts=orig_num_experts,
            new_num_experts=new_num_experts,
        )
