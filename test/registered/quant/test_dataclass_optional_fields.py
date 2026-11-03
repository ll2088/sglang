# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field checks.

Upstream Model-Optimizer PR #3063 fixed a `hasattr` check on dataclass
fields, which is always true, by switching to `is not None`.  These tests
exercise the same pattern with a local dataclass so they do not depend on
any heavy runtime.
"""

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _OptionalFields:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _has_attr_check(obj):
    # The buggy pattern: hasattr is always True for dataclass fields.
    return hasattr(obj, "vocab_embedding") and hasattr(obj, "lm_head")


def _is_not_none_check(obj):
    # The fixed pattern.
    return obj.vocab_embedding is not None and obj.lm_head is not None


@pytest.mark.parametrize(
    ("vocab_embedding", "lm_head"),
    [
        (None, None),
        (object(), None),
        (None, object()),
        (object(), object()),
    ],
)
def test_hasattr_misjudges_dataclass_fields(vocab_embedding, lm_head):
    obj = _OptionalFields(vocab_embedding=vocab_embedding, lm_head=lm_head)
    # hasattr is always True regardless of the field value.
    assert _has_attr_check(obj) is True
    # The fixed check reflects the actual presence of values.
    expected = vocab_embedding is not None and lm_head is not None
    assert _is_not_none_check(obj) is expected
