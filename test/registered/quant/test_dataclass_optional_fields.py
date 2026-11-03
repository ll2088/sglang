# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field handling.

Upstream Model-Optimizer PR #3063 fixed a bug where ``hasattr`` was used
to check dataclass fields that are always present (and therefore always
``True``).  The correct check is ``is not None``.  These tests exercise
the pattern with a local dataclass so they do not depend on heavy runtime
modules.
"""

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _OptionalFields:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _has_attr_bug(obj, name):
    # The buggy pattern: hasattr is always True for dataclass fields.
    return hasattr(obj, name)


def _is_not_none_fix(obj, name):
    # The fixed pattern: check the value, not the attribute.
    return getattr(obj, name, None) is not None


@pytest.mark.parametrize("name", ["vocab_embedding", "lm_head"])
def test_hasattr_is_always_true_for_dataclass_fields(name):
    obj = _OptionalFields()
    assert _has_attr_bug(obj, name) is True
    assert _is_not_none_fix(obj, name) is False


@pytest.mark.parametrize("name", ["vocab_embedding", "lm_head"])
def test_is_not_none_detects_missing_field(name):
    obj = _OptionalFields()
    assert _is_not_none_fix(obj, name) is False
    setattr(obj, name, object())
    assert _is_not_none_fix(obj, name) is True


def test_both_fields_missing_does_not_raise():
    obj = _OptionalFields()
    assert _is_not_none_fix(obj, "vocab_embedding") is False
    assert _is_not_none_fix(obj, "lm_head") is False
