# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field presence checks.

Upstream Model-Optimizer PR #3063 fixed `hasattr` checks on dataclass
fields (always true) by switching to `is not None`.  These tests pin the
same pattern with a local dataclass so they stay dependency-free.
"""

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _OptionalFields:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _has_attr_style(obj, name):
    # The buggy pattern: hasattr is always True for declared dataclass fields.
    return hasattr(obj, name)


def _is_not_none_style(obj, name):
    return getattr(obj, name, None) is not None


@pytest.mark.parametrize("name", ["vocab_embedding", "lm_head"])
def test_hasattr_is_always_true_for_declared_fields(name):
    obj = _OptionalFields()
    assert _has_attr_style(obj, name) is True
    assert _is_not_none_style(obj, name) is False


def test_is_not_none_detects_present_field():
    obj = _OptionalFields(vocab_embedding=object())
    assert _is_not_none_style(obj, "vocab_embedding") is True
    assert _is_not_none_style(obj, "lm_head") is False


def test_missing_field_combination_does_not_crash():
    obj = _OptionalFields(lm_head=object())
    assert _is_not_none_style(obj, "vocab_embedding") is False
    assert _is_not_none_style(obj, "lm_head") is True
