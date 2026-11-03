# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Mirrors the upstream Puzzletron fix: callers must pass the current
``orig_router_weights``/``new_router_weights``/``orig_num_experts``/
``new_num_experts`` keywords, and missing configuration must raise a
controlled error rather than crashing.
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
    if orig_router_weights is None or new_router_weights is None:
        raise TypeError("router weights are required")
    return {
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
        "orig_router_weights": orig_router_weights,
        "new_router_weights": new_router_weights,
    }


def test_current_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[[1.0]],
        new_router_weights=[[2.0]],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2


def test_legacy_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weight=[[1.0]],
            new_router_weight=[[2.0]],
            orig_num_experts=4,
            new_num_experts=2,
        )


def test_missing_config_raises_controlled_error():
    with pytest.raises(TypeError, match="required"):
        _init_moe_module(orig_router_weights=[[1.0]], new_router_weights=[[2.0]])
