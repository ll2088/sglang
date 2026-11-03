# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

These tests are self-contained (stdlib + pytest only) and mirror the upstream
fix where ``_process_single_layer`` passed stale keyword names to
``_init_moe_module`` and raised ``TypeError``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


def _init_moe_module(
    *,
    orig_router_weights: List[float],
    new_router_weights: List[float],
    orig_num_experts: int,
    new_num_experts: int,
) -> Dict[str, Any]:
    if len(orig_router_weights) != orig_num_experts:
        raise ValueError("orig_router_weights length mismatch")
    if len(new_router_weights) != new_num_experts:
        raise ValueError("new_router_weights length mismatch")
    return {
        "orig_router_weights": list(orig_router_weights),
        "new_router_weights": list(new_router_weights),
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def _process_single_layer(
    *,
    orig_router_weights: List[float],
    new_router_weights: List[float],
    orig_num_experts: int,
    new_num_experts: int,
) -> Dict[str, Any]:
    return _init_moe_module(
        orig_router_weights=orig_router_weights,
        new_router_weights=new_router_weights,
        orig_num_experts=orig_num_experts,
        new_num_experts=new_num_experts,
    )


def test_current_keywords_are_accepted() -> None:
    result = _process_single_layer(
        orig_router_weights=[0.1, 0.2, 0.3, 0.4],
        new_router_weights=[0.1, 0.3],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2
    assert result["new_router_weights"] == [0.1, 0.3]


def test_stale_keywords_raise_type_error() -> None:
    with pytest.raises(TypeError):
        _init_moe_module(  # type: ignore[call-arg]
            orig_router_weight=[0.1],
            new_router_weight=[0.1],
            orig_num_experts=1,
            new_num_experts=1,
        )


def test_mismatched_expert_counts_raise_value_error() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        _process_single_layer(
            orig_router_weights=[0.1, 0.2],
            new_router_weights=[0.1],
            orig_num_experts=4,
            new_num_experts=1,
        )
