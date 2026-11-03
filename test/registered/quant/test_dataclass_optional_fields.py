# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field presence checks.

`hasattr` on a dataclass field is always True once the field is declared,
even when its value is None. Code that guards on `hasattr(obj, "field")`
therefore takes the wrong branch for missing values. These tests pin the
`is not None` pattern used by the upstream fix.

Self-contained: stdlib + pytest only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest


@dataclass
class _ModelConfig:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _pad_embedding_lm_head(config: _ModelConfig) -> bool:
    """Mirror of the upstream `pad_embedding_lm_head` guard."""
    if config.vocab_embedding is None or config.lm_head is None:
        return False
    return True


def _hasattr_guard(config: _ModelConfig) -> bool:
    """The buggy pattern: always True for declared dataclass fields."""
    return hasattr(config, "vocab_embedding") and hasattr(config, "lm_head")


@pytest.mark.parametrize(
    ("vocab", "head"),
    [
        (None, None),
        (object(), None),
        (None, object()),
        (object(), object()),
    ],
)
def test_is_not_none_guard_matches_presence(vocab, head):
    config = _ModelConfig(vocab_embedding=vocab, lm_head=head)
    expected = vocab is not None and head is not None
    assert _pad_embedding_lm_head(config) is expected


def test_hasattr_is_always_true_for_declared_fields():
    config = _ModelConfig()
    assert _hasattr_guard(config) is True
    assert _pad_embedding_lm_head(config) is False


def test_missing_field_raises_attribute_error_not_hasattr_true():
    @dataclass
    class _Partial:
        vocab_embedding: Optional[object] = None

    partial = _Partial()
    assert hasattr(partial, "vocab_embedding")
    assert not hasattr(partial, "lm_head")
    with pytest.raises(AttributeError):
        _ = partial.lm_head
