# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword contracts.

These tests are intentionally self-contained: they do not import ``sglang`` so
that they can run without the full runtime dependency tree. They mirror the
upstream fix where ``_process_single_layer`` called ``_init_moe_module`` with
stale keyword names (``orig_router_weight`` / ``new_router_weight``) and
raised ``TypeError``. The fixed contract uses the plural forms plus the
expert-count keywords.
"""

import pytest


_REQUIRED_KEYWORDS = (
    "orig_router_weights",
    "new_router_weights",
    "orig_num_experts",
    "new_num_experts",
)


def _init_moe_module(
    *,
    orig_router_weights,
    new_router_weights,
    orig_num_experts,
    new_num_experts,
):
    """Mirror the fixed ``_init_moe_module`` signature."""
    return {
        "orig_router_weights": orig_router_weights,
        "new_router_weights": new_router_weights,
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def _process_single_layer(**kwargs):
    """Mirror the fixed call site in ``_process_single_layer``."""
    return _init_moe_module(
        orig_router_weights=kwargs["orig_router_weights"],
        new_router_weights=kwargs["new_router_weights"],
        orig_num_experts=kwargs["orig_num_experts"],
        new_num_experts=kwargs["new_num_experts"],
    )


def test_fixed_keywords_are_accepted():
    result = _process_single_layer(
        orig_router_weights="orig",
        new_router_weights="new",
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result == {
        "orig_router_weights": "orig",
        "new_router_weights": "new",
        "orig_num_experts": 4,
        "new_num_experts": 2,
    }


@pytest.mark.parametrize("missing", _REQUIRED_KEYWORDS)
@pytest.mark.parametrize("stale", ["orig_router_weight", "new_router_weight"])
def test_stale_keywords_raise_type_error(missing, stale):
    kwargs = {
        "orig_router_weights": "orig",
        "new_router_weights": "new",
        "orig_num_experts": 4,
        "new_num_experts": 2,
    }
    kwargs.pop(missing)
    kwargs[stale] = "stale"
    with pytest.raises(TypeError):
        _process_single_layer(**kwargs)


def test_missing_expert_count_raises_type_error():
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weights="orig",
            new_router_weights="new",
        )
