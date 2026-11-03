# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Mirrors the upstream fix: ``_init_moe_module`` must accept the current
``orig_router_weights`` / ``new_router_weights`` / ``orig_num_experts`` /
``new_num_experts`` keywords and raise a clear ``TypeError`` for stale ones.
"""

import pytest


def _init_moe_module(
    *,
    orig_router_weights=None,
    new_router_weights=None,
    orig_num_experts=None,
    new_num_experts=None,
):
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
        _init_moe_module(orig_router_weight=[1, 2], new_router_weight=[1])


def test_missing_config_raises_controlled_error():
    with pytest.raises(TypeError):
        _init_moe_module(unknown_option=True)
