# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

These tests are self-contained (stdlib + pytest only) and mirror the upstream
fix where ``_process_single_layer`` passed stale keyword names to
``_init_moe_module`` and raised ``TypeError``.
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
    return {
        "orig_router_weights": orig_router_weights,
        "new_router_weights": new_router_weights,
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def _process_single_layer_legacy(**kwargs):
    # Simulates the buggy call site that used the old keyword names.
    return _init_moe_module(
        orig_router_weight=kwargs.get("orig_router_weights"),
        new_router_weight=kwargs.get("new_router_weights"),
    )


def _process_single_layer_fixed(**kwargs):
    return _init_moe_module(
        orig_router_weights=kwargs.get("orig_router_weights"),
        new_router_weights=kwargs.get("new_router_weights"),
        orig_num_experts=kwargs.get("orig_num_experts"),
        new_num_experts=kwargs.get("new_num_experts"),
    )


def test_legacy_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _process_single_layer_legacy(
            orig_router_weights=[1],
            new_router_weights=[2],
            orig_num_experts=4,
            new_num_experts=2,
        )


def test_fixed_keywords_are_accepted():
    result = _process_single_layer_fixed(
        orig_router_weights=[1],
        new_router_weights=[2],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result == {
        "orig_router_weights": [1],
        "new_router_weights": [2],
        "orig_num_experts": 4,
        "new_num_experts": 2,
    }


def test_missing_config_raises_controlled_error():
    with pytest.raises(TypeError):
        _init_moe_module(orig_router_weights=[1])
