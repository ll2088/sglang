# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field checks.

The upstream fix replaced ``hasattr`` on dataclass fields with an explicit
``is not None`` check, because ``hasattr`` is always true for declared fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _OptionalFields:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _has_required_fields(obj: _OptionalFields) -> bool:
    return obj.vocab_embedding is not None and obj.lm_head is not None


@pytest.mark.parametrize(
    ("vocab_embedding", "lm_head", "expected"),
    [
        (None, None, False),
        (object(), None, False),
        (None, object(), False),
        (object(), object(), True),
    ],
)
def test_optional_dataclass_fields_use_is_not_none(vocab_embedding, lm_head, expected):
    obj = _OptionalFields(vocab_embedding=vocab_embedding, lm_head=lm_head)
    assert _has_required_fields(obj) is expected


def test_hasattr_would_misreport_missing_field():
    obj = _OptionalFields()
    assert hasattr(obj, "vocab_embedding")
    assert obj.vocab_embedding is None
    assert _has_required_fields(obj) is False
