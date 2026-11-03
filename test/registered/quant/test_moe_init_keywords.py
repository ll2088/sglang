# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Mirrors the upstream fix where ``_init_moe_module`` was called with stale
keyword names (``orig_router_weight``/``new_router_weight``) and raised
``TypeError``. These tests are self-contained and do not import ``sglang``.
"""

import pytest


def _init_moe_module(
    *,
    orig_router_weights,
    new_router_weights,
    orig_num_experts,
    new_num_experts,
):
    if orig_num_experts <= 0 or new_num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if len(orig_router_weights) != orig_num_experts:
        raise ValueError("orig_router_weights length mismatch")
    if len(new_router_weights) != new_num_experts:
        raise ValueError("new_router_weights length mismatch")
    return {
        "orig": list(orig_router_weights),
        "new": list(new_router_weights),
    }


def test_current_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[1, 2, 3, 4],
        new_router_weights=[1, 3],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result == {"orig": [1, 2, 3, 4], "new": [1, 3]}


def test_legacy_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weight=[1, 2],
            new_router_weight=[1],
            orig_num_experts=2,
            new_num_experts=1,
        )


@pytest.mark.parametrize(
    ("orig", "new", "orig_n", "new_n"),
    [
        ([], [], 0, 0),
        ([1, 2], [1], 3, 1),
        ([1, 2], [1], 2, 2),
    ],
)
def test_invalid_config_raises_controlled_error(orig, new, orig_n, new_n):
    with pytest.raises(ValueError):
        _init_moe_module(
            orig_router_weights=orig,
            new_router_weights=new,
            orig_num_experts=orig_n,
            new_num_experts=new_n,
        )
