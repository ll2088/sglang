# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field presence checks.

`hasattr` on a dataclass field is always true, so optional fields must be
checked with `is not None`. These tests are self-contained and do not import
sglang.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _ModelConfig:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _pad_embedding_lm_head(config: _ModelConfig) -> bool:
    # Mirrors the fixed upstream logic: check `is not None`, not `hasattr`.
    if config.vocab_embedding is None or config.lm_head is None:
        return False
    return True


@pytest.mark.parametrize(
    ("vocab_embedding", "lm_head", "expected"),
    [
        (None, None, False),
        (object(), None, False),
        (None, object(), False),
        (object(), object(), True),
    ],
)
def test_optional_fields_use_is_not_none(vocab_embedding, lm_head, expected):
    config = _ModelConfig(vocab_embedding=vocab_embedding, lm_head=lm_head)
    assert _pad_embedding_lm_head(config) is expected


def test_hasattr_would_have_misjudged():
    config = _ModelConfig()
    # hasattr is always true for dataclass fields, which is the bug.
    assert hasattr(config, "vocab_embedding")
    assert hasattr(config, "lm_head")
    # The fixed check correctly reports missing values.
    assert _pad_embedding_lm_head(config) is False
