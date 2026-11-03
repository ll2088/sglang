# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE initialization keyword handling.

These tests are self-contained (stdlib + pytest only) and mirror the upstream
fix where ``_process_single_layer`` called ``_init_moe_module`` with stale
keyword names, raising ``TypeError``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


class _MoEConfig:
    def __init__(self, num_experts: int):
        self.num_experts = num_experts


def _init_moe_module(
    *,
    orig_router_weights: List[float],
    new_router_weights: List[float],
    orig_num_experts: int,
    new_num_experts: int,
    **kwargs: Any,
) -> Dict[str, Any]:
    if orig_num_experts <= 0 or new_num_experts <= 0:
        raise ValueError("num_experts must be positive")
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
    orig_config: _MoEConfig,
    new_config: _MoEConfig,
    orig_router_weights: Optional[List[float]] = None,
    new_router_weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if orig_router_weights is None:
        orig_router_weights = [float(i) for i in range(orig_config.num_experts)]
    if new_router_weights is None:
        new_router_weights = [float(i) for i in range(new_config.num_experts)]
    return _init_moe_module(
        orig_router_weights=orig_router_weights,
        new_router_weights=new_router_weights,
        orig_num_experts=orig_config.num_experts,
        new_num_experts=new_config.num_experts,
    )


def test_current_keywords_are_accepted():
    result = _process_single_layer(
        orig_config=_MoEConfig(4), new_config=_MoEConfig(2)
    )
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2
    assert result["orig_router_weights"] == [0.0, 1.0, 2.0, 3.0]
    assert result["new_router_weights"] == [0.0, 1.0]


def test_stale_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weight=[0.0],
            new_router_weight=[0.0],
            orig_num_experts=1,
            new_num_experts=1,
        )


def test_missing_router_weights_are_materialized_per_call():
    first = _process_single_layer(
        orig_config=_MoEConfig(2), new_config=_MoEConfig(2)
    )
    first["orig_router_weights"].append(99.0)
    second = _process_single_layer(
        orig_config=_MoEConfig(2), new_config=_MoEConfig(2)
    )
    assert second["orig_router_weights"] == [0.0, 1.0]


def test_invalid_expert_count_raises_controlled_error():
    with pytest.raises(ValueError, match="num_experts must be positive"):
        _process_single_layer(
            orig_config=_MoEConfig(0), new_config=_MoEConfig(2)
        )


def test_router_weight_length_mismatch_raises_controlled_error():
    with pytest.raises(ValueError, match="orig_router_weights length mismatch"):
        _process_single_layer(
            orig_config=_MoEConfig(4),
            new_config=_MoEConfig(2),
            orig_router_weights=[0.0, 1.0],
        )
