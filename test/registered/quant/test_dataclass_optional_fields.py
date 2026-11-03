# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field presence checks.

Upstream Model-Optimizer PR #3063 fixed `hasattr` checks on dataclass fields
(always true) by switching to `is not None`.  These tests exercise the same
pattern with a local dataclass so they do not depend on heavy runtime.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class _OptionalFields:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _has_attr_style(obj, name):
    # The buggy pattern: hasattr is always True for declared dataclass fields.
    return hasattr(obj, name)


def _is_not_none_style(obj, name):
    # The fixed pattern.
    return getattr(obj, name, None) is not None


def test_hasattr_is_always_true_for_declared_fields():
    obj = _OptionalFields()
    assert _has_attr_style(obj, "vocab_embedding")
    assert _has_attr_style(obj, "lm_head")


def test_is_not_none_detects_missing_fields():
    obj = _OptionalFields()
    assert not _is_not_none_style(obj, "vocab_embedding")
    assert not _is_not_none_style(obj, "lm_head")


def test_is_not_none_detects_present_fields():
    obj = _OptionalFields(vocab_embedding=object(), lm_head=object())
    assert _is_not_none_style(obj, "vocab_embedding")
    assert _is_not_none_style(obj, "lm_head")


def test_partial_fields_are_independent():
    obj = _OptionalFields(vocab_embedding=object())
    assert _is_not_none_style(obj, "vocab_embedding")
    assert not _is_not_none_style(obj, "lm_head")
