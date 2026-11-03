# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field checks.

Mirrors the upstream fix pattern: optional dataclass fields must be checked
with ``is not None`` rather than ``hasattr`` (which is always true for a
declared field).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class _OptionalFields:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _has_required_fields(obj: _OptionalFields) -> bool:
    # The fixed pattern: check the value, not the attribute presence.
    return obj.vocab_embedding is not None and obj.lm_head is not None


def test_hasattr_is_always_true_for_declared_fields():
    obj = _OptionalFields()
    # Documents the bug: hasattr is true even when the value is None.
    assert hasattr(obj, "vocab_embedding")
    assert hasattr(obj, "lm_head")


def test_is_not_none_detects_missing_fields():
    assert not _has_required_fields(_OptionalFields())
    assert not _has_required_fields(_OptionalFields(vocab_embedding=object()))
    assert not _has_required_fields(_OptionalFields(lm_head=object()))


def test_is_not_none_accepts_complete_fields():
    obj = _OptionalFields(vocab_embedding=object(), lm_head=object())
    assert _has_required_fields(obj)
