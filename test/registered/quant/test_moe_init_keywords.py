# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MoE init keyword handling.

Mirrors the upstream fix where ``_init_moe_module`` was called with stale
keyword names, raising ``TypeError``. The helper below models the fixed
contract: it accepts the current keyword names and raises a clear ``TypeError``
when required configuration is missing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest


_REQUIRED_KEYS = (
    "orig_router_weights",
    "new_router_weights",
    "orig_num_experts",
    "new_num_experts",
)


def _init_moe_module(**kwargs: Any) -> Dict[str, Any]:
    missing = [key for key in _REQUIRED_KEYS if key not in kwargs]
    if missing:
        raise TypeError(
            "_init_moe_module() missing required keyword argument(s): "
            + ", ".join(missing)
        )
    return {key: kwargs[key] for key in _REQUIRED_KEYS}


def test_current_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[0, 1],
        new_router_weights=[0],
        orig_num_experts=2,
        new_num_experts=1,
    )
    assert result["orig_num_experts"] == 2
    assert result["new_num_experts"] == 1


def test_legacy_keywords_raise_clear_type_error():
    with pytest.raises(TypeError, match="missing required keyword"):
        _init_moe_module(
            orig_router_weight=[0, 1],
            new_router_weight=[0],
        )


@pytest.mark.parametrize("missing_key", _REQUIRED_KEYS)
def test_each_missing_key_is_reported(missing_key):
    kwargs: Dict[str, Optional[Any]] = {
        "orig_router_weights": [0, 1],
        "new_router_weights": [0],
        "orig_num_experts": 2,
        "new_num_experts": 1,
    }
    kwargs.pop(missing_key)
    with pytest.raises(TypeError, match=missing_key):
        _init_moe_module(**kwargs)
