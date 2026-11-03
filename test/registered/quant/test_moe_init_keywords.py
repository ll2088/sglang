# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

These tests are self-contained: they do not import ``sglang`` so that they can
run without the full runtime dependency tree. They mirror the upstream fix where
``_process_single_layer`` passed stale keyword names to ``_init_moe_module`` and
raised ``TypeError``.
"""

import pytest


def _init_moe_module(
    *,
    orig_router_weights,
    new_router_weights,
    orig_num_experts,
    new_num_experts,
):
    """Mirror the fixed signature: keyword-only, current names."""
    if orig_num_experts <= 0 or new_num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if len(orig_router_weights) != orig_num_experts:
        raise ValueError("orig_router_weights length mismatch")
    if len(new_router_weights) != new_num_experts:
        raise ValueError("new_router_weights length mismatch")
    return {
        "orig_router_weights": orig_router_weights,
        "new_router_weights": new_router_weights,
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def test_current_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[1, 2, 3, 4],
        new_router_weights=[1, 3],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2


def test_stale_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weight=[1, 2, 3, 4],
            new_router_weight=[1, 3],
            orig_num_experts=4,
            new_num_experts=2,
        )


@pytest.mark.parametrize(
    ("orig", "new", "orig_n", "new_n"),
    [
        ([], [1], 0, 1),
        ([1], [], 1, 0),
        ([1, 2], [1], 3, 1),
        ([1], [1, 2], 1, 3),
    ],
)
def test_invalid_configs_raise_controlled_errors(orig, new, orig_n, new_n):
    with pytest.raises(ValueError):
        _init_moe_module(
            orig_router_weights=orig,
            new_router_weights=new,
            orig_num_experts=orig_n,
            new_num_experts=new_n,
        )
