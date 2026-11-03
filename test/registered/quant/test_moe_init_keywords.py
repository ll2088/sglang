# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Mirrors the upstream Puzzletron fix where ``_process_single_layer`` passed
stale keyword names to ``_init_moe_module`` and raised ``TypeError``. The
helper below reproduces the contract: only the current keyword names are
accepted, and missing/extra keywords raise a controlled ``TypeError``.
"""

import pytest


_CURRENT_KEYWORDS = frozenset(
    {
        "orig_router_weights",
        "new_router_weights",
        "orig_num_experts",
        "new_num_experts",
    }
)


def _init_moe_module(**kwargs):
    unknown = set(kwargs) - _CURRENT_KEYWORDS
    if unknown:
        raise TypeError(f"unexpected keyword argument(s): {sorted(unknown)}")
    missing = _CURRENT_KEYWORDS - set(kwargs)
    if missing:
        raise TypeError(f"missing required keyword argument(s): {sorted(missing)}")
    return {key: kwargs[key] for key in _CURRENT_KEYWORDS}


def test_current_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[0, 1],
        new_router_weights=[0],
        orig_num_experts=2,
        new_num_experts=1,
    )
    assert result["orig_num_experts"] == 2
    assert result["new_num_experts"] == 1


@pytest.mark.parametrize(
    "stale_kwargs",
    [
        {"orig_router_weight": [0, 1], "new_router_weight": [0]},
        {"orig_router_weight": [0, 1]},
    ],
)
def test_stale_keywords_raise_type_error(stale_kwargs):
    with pytest.raises(TypeError, match="unexpected keyword"):
        _init_moe_module(
            orig_router_weights=[0, 1],
            new_router_weights=[0],
            orig_num_experts=2,
            new_num_experts=1,
            **stale_kwargs,
        )


def test_missing_keywords_raise_type_error():
    with pytest.raises(TypeError, match="missing required keyword"):
        _init_moe_module(orig_router_weights=[0, 1])
