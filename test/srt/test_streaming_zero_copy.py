"""Tests for the zero-copy streaming token history helpers."""

from __future__ import annotations

import pytest

from sglang.srt.utils.streaming import token_history_len, token_history_view


@pytest.mark.parametrize("sent_count", [0, 1, 3, 5])
def test_view_matches_full_copy_suffix(sent_count: int) -> None:
    history = [10, 20, 30, 40, 50]
    view = token_history_view(history, sent_count)
    assert list(view) == list(history)[sent_count:]


def test_view_does_not_copy_or_mutate_source() -> None:
    history = [1, 2, 3]
    view = token_history_view(history, 1)
    assert list(view) == [2, 3]
    history.append(4)
    assert list(view) == [2, 3, 4]
    assert history == [1, 2, 3, 4]


@pytest.mark.parametrize("history", [None, []])
def test_empty_history(history) -> None:
    assert list(token_history_view(history, 0)) == []
    assert token_history_len(history) == 0


def test_len_matches_sequence() -> None:
    assert token_history_len([1, 2, 3]) == 3
