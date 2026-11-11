# SPDX-License-Identifier: Apache-2.0
"""Helpers for computing perplexity over answer tokens only.

Prompt tokens (system prompt, question, chat-template markup) are nearly
unpredictable, so their log-probs sit in the softmax tail where tiny numeric
differences move them a lot. Restricting the score to answer tokens keeps the
metric focused on what the model is actually asked to predict.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def answer_token_ppl(
    token_logprobs: Sequence[float],
    answer_mask: Sequence[bool],
) -> float:
    """Compute perplexity over the tokens selected by ``answer_mask``.

    Args:
        token_logprobs: Per-token log-probabilities assigned by the model.
        answer_mask: Boolean mask aligned with ``token_logprobs``; ``True``
            marks tokens that belong to the reference answer.

    Returns:
        ``exp(-mean(logprob))`` over the selected tokens.

    Raises:
        ValueError: If the inputs have different lengths or no token is
            selected by the mask.
    """
    if len(token_logprobs) != len(answer_mask):
        raise ValueError(
            "token_logprobs and answer_mask must have the same length, "
            f"got {len(token_logprobs)} and {len(answer_mask)}"
        )
    selected = [lp for lp, keep in zip(token_logprobs, answer_mask) if keep]
    if not selected:
        raise ValueError("answer_mask selects no tokens")
    mean_logprob = sum(selected) / len(selected)
    return math.exp(-mean_logprob)


def build_answer_mask(
    token_ids: Iterable[int],
    answer_token_ids: Iterable[int],
) -> list[bool]:
    """Mark the trailing occurrence of ``answer_token_ids`` in ``token_ids``.

    This is a small, dependency-free helper for tests that need to score only
    the answer span. It returns a mask of the same length as ``token_ids``.
    """
    ids = list(token_ids)
    answer = list(answer_token_ids)
    mask = [False] * len(ids)
    if not answer or len(answer) > len(ids):
        return mask
    start = len(ids) - len(answer)
    if ids[start:] == answer:
        for i in range(start, len(ids)):
            mask[i] = True
    return mask
