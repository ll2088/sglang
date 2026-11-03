# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field checks.

``hasattr`` on a dataclass field is always true, so optional fields must be
checked with ``is not None`` instead. This mirrors the upstream fix in
``trtllm/postprocess.py`` (``pad_embedding_lm_head``).
"""

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _OptionalFields:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _has_required_fields(holder: _OptionalFields) -> bool:
    # Mirrors the fixed pattern: ``is not None``, not ``hasattr``.
    return holder.vocab_embedding is not None and holder.lm_head is not None


def _has_required_fields_via_hasattr(holder: _OptionalFields) -> bool:
    # The buggy pattern: ``hasattr`` is always True for dataclass fields.
    return hasattr(holder, "vocab_embedding") and hasattr(holder, "lm_head")


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
    # ``hasattr`` would always be True for both fields; the fixed check must not.
    assert hasattr(holder, "vocab_embedding")
    assert hasattr(holder, "lm_head")
    assert _has_required_fields(holder) is expected
    # The buggy ``hasattr`` check would always report True; the fixed check
    # must not, otherwise missing optional fields would crash downstream.
    assert _has_required_fields_via_hasattr(holder) is True
