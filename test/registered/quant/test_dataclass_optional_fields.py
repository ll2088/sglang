# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field presence checks.

These tests are self-contained (stdlib + pytest only) and mirror the upstream
fix that replaced ``hasattr`` on dataclass fields with ``is not None`` checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _LayerConfig:
    vocab_embedding: Optional[str] = None
    lm_head: Optional[str] = None


def _pad_embedding_lm_head(layer: _LayerConfig) -> bool:
    # Mirrors the upstream fix: dataclass fields always exist, so ``hasattr``
    # is always True and cannot detect a missing value.
    if layer.vocab_embedding is None or layer.lm_head is None:
        return False
    return True


@pytest.mark.parametrize(
    ("vocab_embedding", "lm_head", "expected"),
    [
        (None, None, False),
        ("embed", None, False),
        (None, "head", False),
        ("embed", "head", True),
    ],
)
def test_optional_dataclass_fields_use_is_not_none(vocab_embedding, lm_head, expected):
    layer = _LayerConfig(vocab_embedding=vocab_embedding, lm_head=lm_head)
    assert _pad_embedding_lm_head(layer) is expected


def test_hasattr_would_have_misjudged_missing_field():
    layer = _LayerConfig()
    # ``hasattr`` is always True for declared dataclass fields, which is the
    # exact bug the upstream fix addressed.
    assert hasattr(layer, "vocab_embedding") is True
    assert layer.vocab_embedding is None
    assert _pad_embedding_lm_head(layer) is False
