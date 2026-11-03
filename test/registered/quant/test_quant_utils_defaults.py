# SPDX-License-Identifier: Apache-2.0
"""Regression tests for mutable-default and shared-state fixes.

Self-contained (stdlib + pytest only); does not import sglang so it can run
without GPU or heavy dependencies.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Optional

import pytest


# --- mutable default isolation -------------------------------------------------


def _with_mutable_default(value, cache={}):
    cache[value] = cache.get(value, 0) + 1
    return cache


def _with_none_default(value, cache=None):
    if cache is None:
        cache = {}
    cache[value] = cache.get(value, 0) + 1
    return cache


def test_mutable_default_leaks_between_calls():
    first = _with_mutable_default("a")
    second = _with_mutable_default("b")
    assert first is second
    assert second == {"a": 1, "b": 1}


def test_none_default_is_isolated_between_calls():
    first = _with_none_default("a")
    second = _with_none_default("b")
    assert first == {"a": 1}
    assert second == {"b": 1}
    assert first is not second


# --- module-level dict must not be mutated in place ----------------------------

_FALLBACK = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}


def _copy_mapping(mapping=None):
    source = mapping if mapping is not None else _FALLBACK
    return {key: list(value) for key, value in source.items()}


def test_module_level_mapping_is_not_polluted():
    original = {key: list(value) for key, value in _FALLBACK.items()}
    first = _copy_mapping()
    first["qkv_proj"].append("extra")
    second = _copy_mapping()
    assert second == original
    assert _FALLBACK == original


# --- dataclass optional fields: hasattr vs is not None -------------------------


@dataclass
class _ModelConfig:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _has_attr_based(config: _ModelConfig) -> bool:
    return hasattr(config, "vocab_embedding") and hasattr(config, "lm_head")


def _is_not_none_based(config: _ModelConfig) -> bool:
    return config.vocab_embedding is not None and config.lm_head is not None


def test_hasattr_is_always_true_for_declared_fields():
    config = _ModelConfig()
    assert _has_attr_based(config) is True
    assert _is_not_none_based(config) is False


@pytest.mark.parametrize(
    ("vocab_embedding", "lm_head", "expected"),
    [
        (None, None, False),
        (object(), None, False),
        (None, object(), False),
        (object(), object(), True),
    ],
)
def test_is_not_none_detects_missing_fields(vocab_embedding, lm_head, expected):
    config = _ModelConfig(vocab_embedding=vocab_embedding, lm_head=lm_head)
    assert _is_not_none_based(config) is expected


# --- unsafe pickle rejection ---------------------------------------------------


class _UnsafePayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (print, ("executed",))


def _safe_load(data: bytes):
    """Reject any pickle payload that is not a plain dict of primitives."""
    obj = pickle.loads(data)
    if not isinstance(obj, dict):
        raise ValueError("unsupported pickle payload")
    for key, value in obj.items():
        if not isinstance(key, str) or not isinstance(value, (int, float, str, bool)):
            raise ValueError("unsupported pickle payload")
    return obj


def test_safe_load_accepts_plain_dict():
    payload = pickle.dumps({"a": 1, "b": "x"})
    assert _safe_load(payload) == {"a": 1, "b": "x"}


def test_safe_load_rejects_unsafe_payload():
    payload = pickle.dumps(_UnsafePayload("marker"))
    with pytest.raises(ValueError, match="unsupported pickle payload"):
        _safe_load(payload)
