import pytest

from sglang.srt.utils.streaming import (
    aligned_logprob_slice,
    new_token_slice,
    token_history_view,
)


@pytest.mark.parametrize("token_ids", [None, [], [1, 2, 3]])
def test_token_history_view_does_not_copy(token_ids):
    view = token_history_view(token_ids)
    if not token_ids:
        assert view == ()
    else:
        assert view is token_ids
        assert list(view) == token_ids


@pytest.mark.parametrize(
    "token_ids,sent,expected",
    [
        (None, 0, ()),
        ([], 0, ()),
        ([1, 2, 3], 0, [1, 2, 3]),
        ([1, 2, 3], 1, [2, 3]),
        ([1, 2, 3], 3, ()),
        ([1, 2, 3], 5, ()),
    ],
)
def test_new_token_slice(token_ids, sent, expected):
    assert list(new_token_slice(token_ids, sent)) == list(expected)


def test_new_token_slice_does_not_mutate_source():
    token_ids = [10, 20, 30]
    new_token_slice(token_ids, 1)
    assert token_ids == [10, 20, 30]


def test_aligned_logprob_slice_matches_token_slice():
    token_ids = [10, 20, 30]
    logprobs = [-0.1, -0.2, -0.3]
    sent = 1
    assert list(new_token_slice(token_ids, sent)) == [20, 30]
    assert list(aligned_logprob_slice(logprobs, sent)) == [-0.2, -0.3]


def test_aligned_logprob_slice_empty():
    assert aligned_logprob_slice(None, 0) == ()
    assert aligned_logprob_slice([], 0) == ()
    assert aligned_logprob_slice([-0.1], 1) == ()
