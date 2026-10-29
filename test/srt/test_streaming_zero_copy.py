"""Unit tests for the zero-copy streaming token history view."""

from sglang.srt.utils.streaming import token_history_len, token_history_view


def test_view_matches_copy_for_all_offsets() -> None:
    history = [10, 20, 21, 30, 31, 32]
    for sent in range(len(history) + 1):
        view = token_history_view(history, sent)
        assert list(view) == list(history)[sent:]


def test_view_does_not_mutate_source() -> None:
    history = [1, 2, 3]
    view = token_history_view(history, 1)
    assert list(view) == [2, 3]
    assert history == [1, 2, 3]


def test_view_handles_none_and_empty() -> None:
    assert token_history_view(None, 0) == ()
    assert token_history_view(None, 5) == ()
    assert token_history_view([], 0) == []
    assert token_history_len(None) == 0
    assert token_history_len([]) == 0
    assert token_history_len([1, 2]) == 2


def test_view_aliases_source_suffix() -> None:
    history = [1, 2, 3, 4]
    view = token_history_view(history, 2)
    history.append(5)
    assert list(view) == [3, 4, 5]
