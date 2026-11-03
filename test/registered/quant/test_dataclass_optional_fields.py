# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field checks.

Upstream replaced ``hasattr`` checks on dataclass fields (always true) with
``is not None`` checks. These tests exercise the pattern in isolation so the
fix can be verified without importing sglang.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _LayerConfig:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _pad_embedding_lm_head(config: _LayerConfig) -> bool:
    # Mirrors the fixed upstream logic: check ``is not None`` rather than
    # ``hasattr`` (which is always true for dataclass fields).
    if config.vocab_embedding is None or config.lm_head is None:
        return False
    return True


@pytest.mark.parametrize(
    ("vocab", "head", "expected"),
    [
        (None, None, False),
        (object(), None, False),
        (None, object(), False),
        (object(), object(), True),
    ],
)
def test_optional_field_presence(vocab, head, expected):
    config = _LayerConfig(vocab_embedding=vocab, lm_head=head)
    assert _pad_embedding_lm_head(config) is expected


def test_hasattr_would_mislead():
    config = _LayerConfig()
    # ``hasattr`` is always true for dataclass fields, which is exactly the
    # bug the upstream fix addresses.
    assert hasattr(config, "vocab_embedding")
    assert _pad_embedding_lm_head(config) is False
