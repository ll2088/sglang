# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the Qwen PPL path under the AITER switch.

This mirrors the upstream fix: when AITER is on, the PPL test must still be
collectable and the answer-token PPL helper must produce a stable value. The
test is intentionally lightweight so it runs without a GPU or model weights;
the heavy generation PPL test lives elsewhere and is exercised by CI on ROCm.
"""

import math
import os
import sys

import pytest

# Import the helper module directly so this test does not require the full
# sglang package (and its heavy dependencies) to be importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PPL_DIR = os.path.abspath(
    os.path.join(
        _HERE,
        "..",
        "..",
        "..",
        "python",
        "sglang",
        "multimodal_gen",
        "test",
        "server",
    )
)
if _PPL_DIR not in sys.path:
    sys.path.insert(0, _PPL_DIR)

from ppl_utils import answer_token_ppl, build_answer_mask  # noqa: E402


def test_qwen_ppl_collects_under_aiter(aiter_config):
    # The fixture must be resolvable from this directory and must select one
    # of the two supported AITER configurations.
    assert aiter_config in {"aiter_off", "aiter_on"}
    assert os.environ.get("SGLANG_USE_AITER") in {"0", "1"}


def test_qwen_answer_token_ppl_is_stable_under_aiter(aiter_config):
    # The answer-token PPL must not depend on the AITER switch: the metric is
    # computed from log-probs, and the switch only changes kernel selection.
    logprobs = [-9.0, -8.5, -0.2, -0.3]
    mask = [False, False, True, True]
    ppl = answer_token_ppl(logprobs, mask)
    assert ppl == pytest.approx(math.exp(0.25))
    assert build_answer_mask([1, 2, 3, 4], [3, 4]) == [False, False, True, True]
