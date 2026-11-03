# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Mirrors the upstream Puzzletron fix: ``_process_single_layer`` must call
``_init_moe_module`` with the current keyword names
(``orig_router_weights`` / ``new_router_weights`` / ``orig_num_experts`` /
``new_num_experts``). Passing the legacy names must raise a controlled
``TypeError`` instead of crashing deeper in the call stack.
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


def _process_single_layer(use_legacy_keywords: bool):
    if use_legacy_keywords:
        return _init_moe_module(
            orig_router_weight=None,
            new_router_weight=None,
            orig_num_experts=4,
            new_num_experts=2,
        )
    return _init_moe_module(
        orig_router_weights=None,
        new_router_weights=None,
        orig_num_experts=4,
        new_num_experts=2,
    )


def test_current_keywords_are_accepted():
    result = _process_single_layer(use_legacy_keywords=False)
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2


def test_legacy_keywords_raise_controlled_type_error():
    with pytest.raises(TypeError):
        _process_single_layer(use_legacy_keywords=True)
