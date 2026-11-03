# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field checks.

Upstream Model-Optimizer PR #3063 replaced `hasattr` on dataclass fields
with `is not None`. `hasattr` is always true for declared dataclass fields,
so a missing value was not detected.
"""

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _ModelConfig:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _pad_embedding_lm_head(config: _ModelConfig) -> bool:
    # Mirrors the fixed upstream logic: check the value, not the attribute.
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


def test_hasattr_would_have_been_wrong():
    config = _ModelConfig()
    # Both attributes exist on the dataclass, so hasattr is always True.
    assert hasattr(config, "vocab_embedding")
    assert hasattr(config, "lm_head")
    # The corrected check still reports the fields as missing.
    assert _pad_embedding_lm_head(config) is False
