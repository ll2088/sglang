from sglang.srt.utils.streaming import (
    aligned_logprob_slice,
    new_token_slice,
    token_history_view,
)


def test_token_history_view_does_not_copy() -> None:
    history = [1, 2, 3]
    view = token_history_view(history)
    assert view is history
    assert token_history_view(None) == ()
    assert token_history_view([]) == ()


def test_new_token_slice_matches_full_copy() -> None:
    history = list(range(10))
    sent = 4
    expected = list(history)[sent:]
    assert list(new_token_slice(history, sent)) == expected
    assert list(new_token_slice(history, 0)) == history
    assert list(new_token_slice(history, len(history))) == []
    assert list(new_token_slice(None, 0)) == []


def test_new_token_slice_does_not_mutate_source() -> None:
    history = [5, 6, 7]
    snapshot = list(history)
    new_token_slice(history, 1)
    assert history == snapshot


def test_logprob_alignment() -> None:
    logprobs = [-0.1, -0.2, -0.3]
    assert list(aligned_logprob_slice(logprobs, 1)) == [-0.2, -0.3]
    assert list(aligned_logprob_slice(logprobs, 0)) == logprobs
    assert list(aligned_logprob_slice(None, 0)) == []
