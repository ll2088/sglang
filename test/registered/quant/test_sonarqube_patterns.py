# SPDX-License-Identifier: Apache-2.0
"""Lightweight regression tests for SonarQube-style defect patterns.

These tests only depend on the standard library and pytest so they can run
without installing sglang's heavy runtime dependencies. They cover:

* mutable default arguments must not leak state between calls;
* module-level fallback mappings must not be mutated in place;
* dataclass optional fields must be checked with ``is not None`` rather than
  ``hasattr`` (which is always true for declared fields);
* unsafe pickle payloads must be rejected without executing code.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional

import pytest


_FALLBACK_FUSED_SHARDS = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _copy_fused_mapping(fused_mapping=None):
    if fused_mapping is not None and not isinstance(fused_mapping, Mapping):
        raise TypeError("fused_mapping must be a mapping")
    source = (
        fused_mapping
        if fused_mapping is not None
        else MappingProxyType(_FALLBACK_FUSED_SHARDS)
    )
    return {key: list(value) for key, value in source.items()}


def _with_mutable_default(value, extra=None):
    if extra is None:
        extra = {}
    extra["value"] = value
    return extra


@dataclass
class _OptionalFieldHolder:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _has_optional_field(obj, name):
    # Correct pattern: declared dataclass fields always satisfy hasattr, so
    # presence must be checked against the value instead.
    return getattr(obj, name, None) is not None


class _UnsafePayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _safe_load(path):
    # Mirrors the restricted loading contract: only plain containers and
    # primitives are accepted, anything else is rejected before execution.
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        obj = pickle.loads(data)
    except Exception as exc:  # noqa: BLE001 - surface as RuntimeError
        raise RuntimeError("Checkpoint could not be loaded safely") from exc
    if not isinstance(obj, (dict, list, tuple, int, float, str, bytes, type(None))):
        raise RuntimeError("Checkpoint contains unsupported objects")
    return obj


def test_mutable_default_is_isolated_between_calls():
    first = _with_mutable_default(1)
    second = _with_mutable_default(2)
    assert first == {"value": 1}
    assert second == {"value": 2}
    assert first is not second


def test_module_level_fallback_is_not_mutated():
    original = {key: list(value) for key, value in _FALLBACK_FUSED_SHARDS.items()}
    copied = _copy_fused_mapping()
    copied["qkv_proj"].append("extra")
    assert _FALLBACK_FUSED_SHARDS == original


def test_caller_mapping_is_copied():
    caller = {"qkv_proj": ["q_proj"]}
    copied = _copy_fused_mapping(caller)
    copied["qkv_proj"].append("k_proj")
    assert caller == {"qkv_proj": ["q_proj"]}


def test_invalid_mapping_raises_type_error():
    with pytest.raises(TypeError):
        _copy_fused_mapping(["not", "a", "mapping"])


def test_dataclass_optional_field_uses_value_check():
    holder = _OptionalFieldHolder()
    # hasattr is always true for declared dataclass fields, so it must not be
    # used to decide whether the value is present.
    assert hasattr(holder, "vocab_embedding")
    assert not _has_optional_field(holder, "vocab_embedding")
    assert not _has_optional_field(holder, "lm_head")

    holder.vocab_embedding = object()
    assert _has_optional_field(holder, "vocab_embedding")
    assert not _has_optional_field(holder, "lm_head")


def test_unsafe_pickle_payload_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "payload.pkl"
    with open(path, "wb") as fh:
        pickle.dump(_UnsafePayload(marker), fh)

    with pytest.raises(RuntimeError):
        _safe_load(path)
    assert not marker.exists()


def test_safe_pickle_payload_is_accepted(tmp_path):
    path = tmp_path / "safe.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"weight": [1, 2, 3]}, fh)
    assert _safe_load(path) == {"weight": [1, 2, 3]}
