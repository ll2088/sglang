# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field checks.

`hasattr` on a dataclass field is always true, so optional fields must be
checked with `is not None` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class _Config:
    vocab_embedding: Optional[str] = None
    lm_head: Optional[str] = None


def _has_embedding(config: _Config) -> bool:
    return config.vocab_embedding is not None


def _has_lm_head(config: _Config) -> bool:
    return config.lm_head is not None


def test_missing_optional_fields_are_detected():
    config = _Config()
    assert not _has_embedding(config)
    assert not _has_lm_head(config)


def test_present_optional_fields_are_detected():
    config = _Config(vocab_embedding="emb", lm_head="head")
    assert _has_embedding(config)
    assert _has_lm_head(config)


def test_partial_fields_are_independent():
    config = _Config(vocab_embedding="emb")
    assert _has_embedding(config)
    assert not _has_lm_head(config)
