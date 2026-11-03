# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field presence checks.

`hasattr` on a dataclass field is always True once the field is declared, so
optional fields must be checked with `is not None`. These tests are
self-contained (stdlib + pytest only) and do not import sglang.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest


@dataclass
class _OptionalFields:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None
    extra: dict = field(default_factory=dict)


def _has_attr_based(obj, name):
    # The buggy pattern: always True for declared dataclass fields.
    return hasattr(obj, name)


def _is_not_none_based(obj, name):
    # The fixed pattern.
    return getattr(obj, name, None) is not None


@pytest.mark.parametrize("name", ["vocab_embedding", "lm_head"])
def test_hasattr_is_always_true_for_declared_fields(name):
    obj = _OptionalFields()
    assert _has_attr_based(obj, name) is True
    assert _is_not_none_based(obj, name) is False


@pytest.mark.parametrize(
    ("vocab_embedding", "lm_head"),
    [(None, None), (object(), None), (None, object()), (object(), object())],
)
def test_is_not_none_distinguishes_missing_fields(vocab_embedding, lm_head):
    obj = _OptionalFields(vocab_embedding=vocab_embedding, lm_head=lm_head)
    assert _is_not_none_based(obj, "vocab_embedding") is (vocab_embedding is not None)
    assert _is_not_none_based(obj, "lm_head") is (lm_head is not None)


def test_dataclass_default_factory_is_isolated():
    first = _OptionalFields()
    first.extra["k"] = 1
    second = _OptionalFields()
    assert second.extra == {}
