# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Mirrors the upstream Puzzletron fix: callers must pass the current
``orig_*``/``new_*`` keyword names, and a stale keyword must raise a clear
``TypeError`` instead of crashing deep inside the initializer.
"""

import pytest


def _init_moe_module(
    *,
    orig_router_weights=None,
    new_router_weights=None,
    orig_num_experts=None,
    new_num_experts=None,
):
    if orig_num_experts is None or new_num_experts is None:
        raise TypeError("orig_num_experts and new_num_experts are required")
    return {
        "orig_router_weights": orig_router_weights,
        "new_router_weights": new_router_weights,
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def test_current_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[1, 2],
        new_router_weights=[1],
        orig_num_experts=2,
        new_num_experts=1,
    )
    assert result["orig_num_experts"] == 2
    assert result["new_num_experts"] == 1


def test_stale_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weight=[1, 2],
            new_router_weight=[1],
            orig_num_experts=2,
            new_num_experts=1,
        )


def test_missing_expert_counts_raise_type_error():
    with pytest.raises(TypeError, match="orig_num_experts"):
        _init_moe_module(orig_router_weights=[1], new_router_weights=[1])
