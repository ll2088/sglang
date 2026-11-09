# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Byte-equivalence, fallback and microbenchmark tests for the plain-delta
completion stream frame template.

These tests only touch the protocol models and the pure helpers in
``serving_completions`` so they run without a tokenizer or engine.
"""

import json
import statistics
import time

import pytest

from sglang.srt.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)
from sglang.srt.entrypoints.openai.serving_completions import (
    _can_use_delta_template,
    _delta_frame_template,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CREATED = 1706000000
REQUEST_ID = "cmpl-abc123"
MODEL = "meta-llama/Llama-3.1-8B"


def model_frame(request_id: str, model_name: str, index: int, text: str) -> str:
    chunk = CompletionStreamResponse(
        id=request_id,
        created=CREATED,
        object="text_completion",
        choices=[
            CompletionResponseStreamChoice(
                index=index,
                text=text,
                logprobs=None,
                finish_reason=None,
                matched_stop=None,
                token_ids=None,
                prompt_token_ids=None,
            )
        ],
        model=model_name,
    )
    return f"data: {chunk.model_dump_json()}\n\n"


def template_frame(template, index: int, text: str) -> str:
    prefix, infix, suffix = template
    return (
        f"{prefix}{index}{infix}"
        f"{json.dumps(text, ensure_ascii=False)}"
        f"{suffix}"
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello, world",
        ' "quoted" and \\backslashed\\ ',
        "line\nbreaks\r\ttabs",
        "\x00\x01\x1f\x7f",
        "unicode é 日本語 🚀 \U0001d549\U0010ffff",
    ],
)
@pytest.mark.parametrize(
    ("request_id", "model_name", "index"),
    [
        (REQUEST_ID, MODEL, 0),
        (REQUEST_ID, MODEL, 7),
        ('id "with" quotes\\', "модель 🎯", 3),
    ],
)
def test_template_frame_matches_model_frame(request_id, model_name, index, text):
    template = _delta_frame_template(request_id, CREATED, model_name)
    assert template is not None
    assert template_frame(template, index, text) == model_frame(
        request_id, model_name, index, text
    )


def test_template_is_none_when_sentinels_are_ambiguous():
    assert _delta_frame_template("cmpl-987654321", CREATED, "m") is None


def test_first_middle_last_frames_are_byte_identical():
    template = _delta_frame_template(REQUEST_ID, CREATED, MODEL)
    assert template is not None
    frames = [("first", 0), (" middle", 0), (" last", 0)]
    for text, index in frames:
        assert template_frame(template, index, text) == model_frame(
            REQUEST_ID, MODEL, index, text
        )


def test_frame_keeps_sse_prefix_and_terminator():
    template = _delta_frame_template(REQUEST_ID, CREATED, MODEL)
    assert template is not None
    frame = template_frame(template, 0, "x")
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert frame == model_frame(REQUEST_ID, MODEL, 0, "x")


def _request(**kwargs) -> CompletionRequest:
    return CompletionRequest(model=MODEL, prompt="hi", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"stream_options": {"include_usage": True}},
    ],
)
def test_plain_requests_use_template(kwargs):
    assert _can_use_delta_template(_request(**kwargs), False) is True


@pytest.mark.parametrize(
    ("kwargs", "continuous_usage"),
    [
        ({"echo": True}, False),
        ({"logprobs": 1}, False),
        ({"return_token_ids": True}, False),
        ({"return_spec_tokens_details": True}, False),
        ({}, True),
    ],
)
def test_non_plain_requests_fall_back(kwargs, continuous_usage):
    assert _can_use_delta_template(_request(**kwargs), continuous_usage) is False


def test_fallback_frame_keeps_all_fields():
    choice = CompletionResponseStreamChoice(
        index=0,
        text="done",
        logprobs=None,
        finish_reason="stop",
        matched_stop="done",
        token_ids=[1, 2],
        prompt_token_ids=[3],
    )
    chunk = CompletionStreamResponse(
        id=REQUEST_ID,
        created=CREATED,
        object="text_completion",
        choices=[choice],
        model=MODEL,
    )
    payload = json.loads(chunk.model_dump_json())
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["matched_stop"] == "done"
    assert payload["choices"][0]["token_ids"] == [1, 2]
    assert payload["choices"][0]["prompt_token_ids"] == [3]


def test_multi_choice_frames_match_model():
    template = _delta_frame_template(REQUEST_ID, CREATED, MODEL)
    assert template is not None
    for index in range(3):
        assert template_frame(template, index, f"c{index}") == model_frame(
            REQUEST_ID, MODEL, index, f"c{index}"
        )


def _median_render_us(render, rounds=5, iters=2000) -> float:
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        for i in range(iters):
            render(i)
        samples.append((time.perf_counter() - start) / iters * 1e6)
    return statistics.median(samples)


def test_microbench_template_not_slower_than_model():
    template = _delta_frame_template(REQUEST_ID, CREATED, MODEL)
    assert template is not None

    def render_template(i):
        return template_frame(template, 0, f"tok{i}")

    def render_model(i):
        return model_frame(REQUEST_ID, MODEL, 0, f"tok{i}")

    assert render_template(0) == render_model(0)
    template_us = _median_render_us(render_template)
    model_us = _median_render_us(render_model)
    print(
        f"template={template_us:.3f}us model={model_us:.3f}us "
        f"speedup={model_us / template_us:.2f}x"
    )
    assert template_us <= model_us * 1.10
