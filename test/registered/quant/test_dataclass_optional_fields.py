# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field presence checks.

Upstream Model-Optimizer PR #3063 replaced `hasattr` checks on dataclass
fields with `is not None` checks, because `hasattr` is always true for a
declared dataclass field even when its value is None.
"""

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _OptionalFields:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _hasattr_check(obj, name):
    # The buggy pattern: always true for declared dataclass fields.
    return hasattr(obj, name)


def _is_not_none_check(obj, name):
    # The fixed pattern.
    return getattr(obj, name, None) is not None


@pytest.mark.parametrize("name", ["vocab_embedding", "lm_head"])
def test_hasattr_is_always_true_for_declared_fields(name):
    obj = _OptionalFields()
    assert _hasattr_check(obj, name) is True
    assert _is_not_none_check(obj, name) is False


@pytest.mark.parametrize("name", ["vocab_embedding", "lm_head"])
def test_is_not_none_detects_missing_field(name):
    obj = _OptionalFields()
    assert _is_not_none_check(obj, name) is False
    setattr(obj, name, object())
    assert _is_not_none_check(obj, name) is True


def test_partial_fields_are_independent():
    obj = _OptionalFields(vocab_embedding=object())
    assert _is_not_none_check(obj, "vocab_embedding") is True
    assert _is_not_none_check(obj, "lm_head") is False
