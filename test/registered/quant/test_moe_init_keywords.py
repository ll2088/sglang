# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

These tests are self-contained (stdlib + pytest only) and mirror the upstream
fix that replaced stale ``orig_router_weight``/``new_router_weight`` keywords
with the current plural keyword names.
"""

from __future__ import annotations

import pytest


def _init_moe_module(
    *,
    orig_router_weights=None,
    new_router_weights=None,
    orig_num_experts=None,
    new_num_experts=None,
):
    if orig_router_weights is None or new_router_weights is None:
        raise TypeError("router weights are required")
    if orig_num_experts is None or new_num_experts is None:
        raise TypeError("expert counts are required")
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


def test_legacy_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weight=[1, 2],
            new_router_weight=[1],
            orig_num_experts=2,
            new_num_experts=1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"new_router_weights": [1], "orig_num_experts": 2, "new_num_experts": 1},
        {"orig_router_weights": [1, 2], "orig_num_experts": 2, "new_num_experts": 1},
        {"orig_router_weights": [1, 2], "new_router_weights": [1], "new_num_experts": 1},
    ],
)
def test_missing_required_keywords_raise_type_error(kwargs):
    with pytest.raises(TypeError):
        _init_moe_module(**kwargs)
