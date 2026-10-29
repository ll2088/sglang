import pytest

from sglang.srt.utils.streaming import (
    aligned_logprob_slice,
    new_token_slice,
    token_history_view,
)


@pytest.mark.parametrize("token_ids", [None, [], [1], [1, 2, 3, 4]])
def test_token_history_view_matches_copy(token_ids):
    view = token_history_view(token_ids)
    expected = list(token_ids) if token_ids else []
    assert list(view) == expected


def test_token_history_view_does_not_copy_list():
    token_ids = [1, 2, 3]
    view = token_history_view(token_ids)
    assert view is token_ids


@pytest.mark.parametrize(
    "token_ids,sent_count,expected",
    [
        (None, 0, []),
        ([], 0, []),
        ([1, 2, 3], 0, [1, 2, 3]),
        ([1, 2, 3], 1, [2, 3]),
        ([1, 2, 3], 3, []),
        ([1, 2, 3], 5, []),
    ],
)
def test_new_token_slice_matches_full_copy(token_ids, sent_count, expected):
    assert list(new_token_slice(token_ids, sent_count)) == expected


def test_new_token_slice_does_not_mutate_source():
    token_ids = [10, 20, 30]
    new_token_slice(token_ids, 1)
    assert token_ids == [10, 20, 30]


def test_aligned_logprob_slice_matches_token_slice():
    token_ids = [1, 2, 3, 4]
    logprobs = [-0.1, -0.2, -0.3, -0.4]
    for sent in range(0, len(token_ids) + 2):
        tokens = list(new_token_slice(token_ids, sent))
        aligned = list(aligned_logprob_slice(logprobs, sent))
        assert len(tokens) == len(aligned)
        assert aligned == logprobs[sent:] if sent < len(logprobs) else aligned == []


def test_streaming_deltas_are_consistent_with_full_history():
    token_ids = [5, 6, 7, 8, 9]
    logprobs = [-0.5, -0.6, -0.7, -0.8, -0.9]
    sent = 0
    collected_tokens = []
    collected_logprobs = []
    while True:
        delta = list(new_token_slice(token_ids, sent))
        delta_lp = list(aligned_logprob_slice(logprobs, sent))
        if not delta:
            break
        collected_tokens.extend(delta)
        collected_logprobs.extend(delta_lp)
        sent += len(delta)
    assert collected_tokens == token_ids
    assert collected_logprobs == logprobs
