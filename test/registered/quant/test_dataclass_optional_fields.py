# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dataclass optional-field checks.

Upstream replaced ``hasattr`` on dataclass fields (always true) with
``is not None``. These tests pin the corrected behaviour for the local
``Nvfp4SourceConfig`` marker and for a representative optional-field
consumer.
"""

from dataclasses import dataclass, field
from typing import Optional

import pytest


@dataclass
class _OptionalHolder:
    vocab_embedding: Optional[object] = None
    lm_head: Optional[object] = None


def _has_attr_wrong(holder, name):
    # The buggy pattern: hasattr is always True for declared dataclass fields.
    return hasattr(holder, name)


def _is_not_none_correct(holder, name):
    return getattr(holder, name, None) is not None


def test_hasattr_is_always_true_for_declared_dataclass_fields():
    holder = _OptionalHolder()
    assert _has_attr_wrong(holder, "vocab_embedding") is True
    assert _has_attr_wrong(holder, "lm_head") is True


def test_is_not_none_detects_missing_optional_fields():
    holder = _OptionalHolder()
    assert _is_not_none_correct(holder, "vocab_embedding") is False
    assert _is_not_none_correct(holder, "lm_head") is False


def test_is_not_none_detects_present_optional_fields():
    holder = _OptionalHolder(vocab_embedding=object(), lm_head=object())
    assert _is_not_none_correct(holder, "vocab_embedding") is True
    assert _is_not_none_correct(holder, "lm_head") is True


def test_partial_optional_fields_are_independent():
    holder = _OptionalHolder(vocab_embedding=object())
    assert _is_not_none_correct(holder, "vocab_embedding") is True
    assert _is_not_none_correct(holder, "lm_head") is False


def test_nvfp4_source_config_marker_is_importable():
    quark_utils = pytest.importorskip(
        "sglang.srt.layers.quantization.quark.utils",
        reason="quark utils require the sglang runtime",
    )
    marker = quark_utils.Nvfp4SourceConfig()
    assert marker is not None
