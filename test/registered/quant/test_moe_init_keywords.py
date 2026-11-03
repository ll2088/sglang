# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Mirrors the upstream Puzzletron fix: the init helper must accept the current
keyword names and raise a clear ``TypeError`` for stale ones instead of
crashing deep inside the call.
"""

import pytest


CURRENT_KEYWORDS = (
    "orig_router_weights",
    "new_router_weights",
    "orig_num_experts",
    "new_num_experts",
)

LEGACY_KEYWORDS = ("orig_router_weight", "new_router_weight")


def init_moe_module(**kwargs):
    missing = [name for name in CURRENT_KEYWORDS if name not in kwargs]
    if missing:
        raise TypeError(
            "init_moe_module() missing required keyword arguments: "
            + ", ".join(missing)
        )
    return {name: kwargs[name] for name in CURRENT_KEYWORDS}


def test_current_keywords_are_accepted():
    result = init_moe_module(
        orig_router_weights=[0.1],
        new_router_weights=[0.2],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2


@pytest.mark.parametrize("legacy", LEGACY_KEYWORDS)
def test_legacy_keywords_raise_clear_type_error(legacy):
    kwargs = {
        "orig_router_weights": [0.1],
        "new_router_weights": [0.2],
        "orig_num_experts": 4,
        "new_num_experts": 2,
    }
    kwargs[legacy] = [0.3]
    with pytest.raises(TypeError, match="missing required keyword arguments"):
        init_moe_module(**kwargs)


def test_missing_config_raises_type_error():
    with pytest.raises(TypeError, match="missing required keyword arguments"):
        init_moe_module(orig_router_weights=[0.1])
