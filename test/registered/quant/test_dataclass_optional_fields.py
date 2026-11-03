# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field checks.

Upstream Model-Optimizer PR #3063 fixed a bug where `hasattr` was used to
check dataclass fields that are always present (and therefore always truthy),
so a missing/None value was never detected.  These tests exercise the same
pattern with a local dataclass so they do not depend on any heavy runtime.
"""

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _OptionalFields:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _has_attr_bug(obj, name):
    # The buggy pattern: hasattr is always True for declared dataclass fields.
    return hasattr(obj, name)


def _is_not_none_fix(obj, name):
    # The fixed pattern: check the value, not the attribute existence.
    return getattr(obj, name, None) is not None


@pytest.mark.parametrize("field", ["vocab_embedding", "lm_head"])
def test_hasattr_is_always_true_for_dataclass_fields(field):
    obj = _OptionalFields()
    assert _has_attr_bug(obj, field) is True
    assert _is_not_none_fix(obj, field) is False


@pytest.mark.parametrize("field", ["vocab_embedding", "lm_head"])
def test_is_not_none_detects_present_value(field):
    obj = _OptionalFields(**{field: object()})
    assert _is_not_none_fix(obj, field) is True


def test_missing_field_does_not_crash_with_getattr_default():
    obj = _OptionalFields()
    assert getattr(obj, "does_not_exist", None) is None
