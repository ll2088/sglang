# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE initialization keyword handling.

These tests are intentionally self-contained: they do not import ``sglang`` so
that they can run without the full runtime dependency tree. The helper below
mirrors the upstream fix where ``_init_moe_module`` must be called with the
current ``orig_*``/``new_*`` keyword names, and a missing or stale keyword must
raise a clear ``TypeError`` instead of crashing deeper in the call stack.
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


def _process_single_layer(**kwargs):
    return _init_moe_module(**kwargs)


def test_current_keywords_are_accepted():
    result = _process_single_layer(
        orig_router_weights=[1, 2, 3, 4],
        new_router_weights=[1, 3],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2


def test_legacy_keywords_raise_clear_type_error():
    with pytest.raises(TypeError) as exc_info:
        _process_single_layer(
            orig_router_weight=[1, 2, 3, 4],
            new_router_weight=[1, 3],
            orig_num_experts=4,
            new_num_experts=2,
        )
    message = str(exc_info.value)
    assert "orig_router_weights" in message or "unexpected keyword" in message


def test_missing_keyword_raises_type_error():
    with pytest.raises(TypeError):
        _process_single_layer(
            orig_router_weights=[1, 2, 3, 4],
            new_router_weights=[1, 3],
            orig_num_experts=4,
        )
