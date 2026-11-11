# SPDX-License-Identifier: Apache-2.0

import math
import os
import sys

import pytest

# Import the helper module directly so this unit test does not require the
# full sglang package (and its heavy dependencies) to be importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ppl_utils import answer_token_ppl, build_answer_mask  # noqa: E402


def test_answer_token_ppl_ignores_prompt_tokens():
    # Prompt tokens carry noisy, near-zero log-probs; answer tokens are
    # confident. Scoring only the answer must not be dragged down by prompt.
    logprobs = [-12.0, -11.5, -0.1, -0.2]
    mask = [False, False, True, True]
    ppl = answer_token_ppl(logprobs, mask)
    assert ppl == pytest.approx(math.exp(0.15))


def test_answer_token_ppl_all_tokens_matches_manual():
    logprobs = [-0.5, -1.0]
    mask = [True, True]
    ppl = answer_token_ppl(logprobs, mask)
    assert ppl == pytest.approx(math.exp(0.75))


def test_answer_token_ppl_rejects_empty_mask():
    with pytest.raises(ValueError):
        answer_token_ppl([-0.1, -0.2], [False, False])


def test_answer_token_ppl_rejects_length_mismatch():
    with pytest.raises(ValueError):
        answer_token_ppl([-0.1], [True, False])


def test_build_answer_mask_marks_trailing_answer():
    mask = build_answer_mask([1, 2, 3, 4], [3, 4])
    assert mask == [False, False, True, True]


def test_build_answer_mask_returns_all_false_when_absent():
    mask = build_answer_mask([1, 2, 3], [9, 9])
    assert mask == [False, False, False]


def test_answer_token_ppl_prompt_vs_answer_comparison():
    # Same model output, two scoring scopes. Prompt tokens are noisy and
    # near-zero log-prob; answer tokens are confident. Scoring only the
    # answer must be strictly better (lower PPL) than scoring everything.
    logprobs = [-12.0, -11.5, -0.1, -0.2]
    all_tokens = answer_token_ppl(logprobs, [True, True, True, True])
    answer_only = answer_token_ppl(logprobs, [False, False, True, True])
    assert answer_only < all_tokens
    assert answer_only == pytest.approx(math.exp(0.15))


def test_answer_token_ppl_single_answer_token():
    ppl = answer_token_ppl([-5.0, -0.5], [False, True])
    assert ppl == pytest.approx(math.exp(0.5))
