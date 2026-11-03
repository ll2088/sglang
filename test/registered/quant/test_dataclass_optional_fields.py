# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field checks.

`hasattr` on a dataclass field is always true, so optional fields must be
checked with `is not None` instead.
"""

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _OptionalFields:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _has_required_fields(holder: _OptionalFields) -> bool:
    # Mirrors the fixed pattern: `is not None`, not `hasattr`.
    return holder.vocab_embedding is not None and holder.lm_head is not None


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
    holder = _OptionalFields(vocab_embedding=vocab_embedding, lm_head=lm_head)
    # `hasattr` would always be True for both fields; the fixed check must not.
    assert hasattr(holder, "vocab_embedding")
    assert hasattr(holder, "lm_head")
    assert _has_required_fields(holder) is expected
