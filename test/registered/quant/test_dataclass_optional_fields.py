# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field presence checks.

Mirrors the upstream fix that replaced ``hasattr`` on dataclass fields with
``is not None`` checks, since ``hasattr`` is always true for declared fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _ModelConfig:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _hasattr_based(config: _ModelConfig) -> bool:
    # The buggy pattern: always true for declared dataclass fields.
    return hasattr(config, "vocab_embedding") and hasattr(config, "lm_head")


def _is_not_none_based(config: _ModelConfig) -> bool:
    # The fixed pattern.
    return config.vocab_embedding is not None and config.lm_head is not None


@pytest.mark.parametrize(
    ("vocab_embedding", "lm_head"),
    [
        (None, None),
        (object(), None),
        (None, object()),
        (object(), object()),
    ],
)
def test_optional_field_presence(vocab_embedding, lm_head):
    config = _ModelConfig(vocab_embedding=vocab_embedding, lm_head=lm_head)
    expected = vocab_embedding is not None and lm_head is not None
    assert _is_not_none_based(config) is expected


def test_hasattr_is_always_true_for_declared_fields():
    config = _ModelConfig()
    assert _hasattr_based(config) is True
    assert _is_not_none_based(config) is False
