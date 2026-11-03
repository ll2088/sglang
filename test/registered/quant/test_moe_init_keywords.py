# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Upstream Model-Optimizer PR #3063 fixed a TypeError caused by calling
`_init_moe_module` with stale keyword names.  These tests exercise the same
pattern with a local helper so they do not depend on any heavy runtime.
"""

import pytest


def _init_moe_module(*, orig_router_weights, new_router_weights,
                     orig_num_experts, new_num_experts):
    return {
        "orig_router_weights": orig_router_weights,
        "new_router_weights": new_router_weights,
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def test_old_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weight=None,
            new_router_weight=None,
            orig_num_experts=4,
            new_num_experts=2,
        )


def test_new_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[1, 2, 3, 4],
        new_router_weights=[1, 3],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2
    assert result["new_router_weights"] == [1, 3]
