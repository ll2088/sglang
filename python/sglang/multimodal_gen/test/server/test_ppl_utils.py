# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from sglang.multimodal_gen.test.server.ppl_utils import (
    answer_token_ppl,
    build_answer_mask,
)


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
