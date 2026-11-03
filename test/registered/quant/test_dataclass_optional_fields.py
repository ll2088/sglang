# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field presence checks.

`hasattr` on a dataclass field is always True once the field is declared,
even when its value is None. Optional fields must be checked with
`is not None` so that a missing value is handled instead of crashing.

Self-contained (stdlib + pytest only); does not import sglang.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _ModelConfig:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _has_attr_based(config: _ModelConfig) -> bool:
    # The buggy pattern: hasattr is always True for declared dataclass fields.
    return hasattr(config, "vocab_embedding") and hasattr(config, "lm_head")


def _is_not_none_based(config: _ModelConfig) -> bool:
    return config.vocab_embedding is not None and config.lm_head is not None


def test_hasattr_is_always_true_for_declared_fields():
    config = _ModelConfig()
    assert _has_attr_based(config) is True
    assert _is_not_none_based(config) is False


@pytest.mark.parametrize(
    ("vocab_embedding", "lm_head", "expected"),
    [
        (None, None, False),
        (object(), None, False),
        (None, object(), False),
        (object(), object(), True),
    ],
)
def test_is_not_none_detects_missing_fields(vocab_embedding, lm_head, expected):
    config = _ModelConfig(vocab_embedding=vocab_embedding, lm_head=lm_head)
    assert _is_not_none_based(config) is expected


def test_missing_field_does_not_crash_with_is_not_none():
    config = _ModelConfig()
    # The fixed check must not raise even when both optional fields are unset.
    assert _is_not_none_based(config) is False
