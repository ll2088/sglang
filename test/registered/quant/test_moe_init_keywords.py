# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

These tests are self-contained and do not import ``sglang``. They mirror
the upstream fix where ``_process_single_layer`` passed stale keyword
names to ``_init_moe_module`` and raised ``TypeError``.
"""

import pytest


def _init_moe_module(
    *,
    orig_router_weights,
    new_router_weights,
    orig_num_experts,
    new_num_experts,
):
    return {
        "orig_router_weights": orig_router_weights,
        "new_router_weights": new_router_weights,
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def _process_single_layer(*, orig_router_weights, new_router_weights,
                          orig_num_experts, new_num_experts):
    return _init_moe_module(
        orig_router_weights=orig_router_weights,
        new_router_weights=new_router_weights,
        orig_num_experts=orig_num_experts,
        new_num_experts=new_num_experts,
    )


def test_current_keywords_are_accepted():
    result = _process_single_layer(
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


def test_missing_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _process_single_layer(orig_router_weights=[1, 2])
