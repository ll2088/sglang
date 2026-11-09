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

# This module must be collectable with only pydantic + pytest installed.
# It never imports the `sglang` package top level (whose __init__ pulls in
# the full runtime); instead it loads protocol.py and serving_completions.py
# from source via importlib with lightweight package stubs.


def _install_minimal_stubs() -> None:
    """No-op kept for readability: the helpers are loaded from source below,
    so no runtime module stubs are needed."""
    return


def _load_template_helpers():
    """Import the pure helpers, stubbing heavy runtime deps if needed.

    The helpers under test only need the protocol models and ``json``; the
    serving module pulls in the full sglang runtime. When that import fails
    (e.g. minimal CI env), load the module source with a stub package so the
    tests still execute instead of being skipped.
    """
    _install_minimal_stubs()
    import importlib.util
    import sys
    import types
    from pathlib import Path

    # If the real package is importable, prefer it: it exercises the actual
    # module wiring. Fall back to source loading only when the runtime is
    # unavailable (minimal CI env with just pydantic + pytest).
    try:
        from sglang.srt.entrypoints.openai import serving_completions as _real

        return (
            _real.CompletionRequest,
            _real.CompletionResponseStreamChoice,
            _real.CompletionStreamResponse,
            _real._can_use_delta_template,
            _real._delta_frame_template,
        )
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[5]
    pkg_root = repo_root / "python" / "sglang"
    protocol_path = pkg_root / "srt" / "entrypoints" / "openai" / "protocol.py"
    serving_path = (
        pkg_root / "srt" / "entrypoints" / "openai" / "serving_completions.py"
    )

    def _ensure_pkg(name: str, path: Path) -> None:
        if name in sys.modules:
            return
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod

    for name, path in (
        ("sglang", pkg_root),
        ("sglang.srt", pkg_root / "srt"),
        ("sglang.srt.entrypoints", pkg_root / "srt" / "entrypoints"),
        ("sglang.srt.entrypoints.openai", pkg_root / "srt" / "entrypoints" / "openai"),
    ):
        _ensure_pkg(name, path)

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    protocol = _load("sglang.srt.entrypoints.openai.protocol", protocol_path)
    serving = _load(
        "sglang.srt.entrypoints.openai.serving_completions", serving_path
    )
    # Guard against accidentally importing the real package top level, which
    # would drag in the full runtime dependency chain during collection.
    assert "sglang" in sys.modules
    assert getattr(sys.modules["sglang"], "__file__", None) is None, (
        "sglang package top level must not be imported by these tests"
    )
    return (
        protocol.CompletionRequest,
        protocol.CompletionResponseStreamChoice,
        protocol.CompletionStreamResponse,
        serving._can_use_delta_template,
        serving._delta_frame_template,
    )


(
    CompletionRequest,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    _can_use_delta_template,
    _delta_frame_template,
) = _load_template_helpers()

try:
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=5, suite="base-a-test-cpu")
except ImportError:  # pragma: no cover - CI registry is optional locally
    pass

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
        ({"return_spec_tokens_details": False}, False),
        ({}, True),
    ],
)
def test_non_plain_requests_fall_back(kwargs, continuous_usage):
    assert _can_use_delta_template(_request(**kwargs), continuous_usage) is False


def test_spec_tokens_details_forces_fallback_branch():
    """return_spec_tokens_details must never take the template path."""
    request = _request(return_spec_tokens_details=True)
    assert _can_use_delta_template(request, False) is False
    # The model path still serializes the trailing sglext chunk correctly.
    chunk = CompletionStreamResponse(
        id=REQUEST_ID,
        created=CREATED,
        object="text_completion",
        choices=[],
        model=MODEL,
    )
    payload = json.loads(chunk.model_dump_json())
    assert payload["choices"] == []


def test_fallback_frame_keeps_all_fields():
    # Non-plain requests must never take the template path.
    for kwargs in (
        {"echo": True},
        {"logprobs": 1},
        {"return_token_ids": True},
        {"return_spec_tokens_details": True},
    ):
        assert _can_use_delta_template(_request(**kwargs), False) is False
    assert _can_use_delta_template(_request(), True) is False

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


def test_finish_reason_frame_uses_model_path():
    # Terminal chunks carry finish_reason/matched_stop and must be rendered by
    # the model serializer, not the plain-delta template.
    choice = CompletionResponseStreamChoice(
        index=0,
        text="",
        logprobs=None,
        finish_reason="stop",
        matched_stop="</s>",
        token_ids=None,
        prompt_token_ids=None,
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
    assert payload["choices"][0]["matched_stop"] == "</s>"


def _median_render_us(render, rounds=3, iters=2000) -> float:
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
    # Three independent runs, each taking the median of its rounds; report the
    # median of the run medians so the CI log carries a stable number.
    template_runs = [_median_render_us(render_template) for _ in range(3)]
    model_runs = [_median_render_us(render_model) for _ in range(3)]
    template_us = statistics.median(template_runs)
    model_us = statistics.median(model_runs)
    print(
        "microbench template_us=%s model_us=%s speedup=%.2fx"
        % (
            [round(v, 3) for v in template_runs],
            [round(v, 3) for v in model_runs],
            model_us / template_us,
        )
    )
    assert template_us <= model_us * 1.10


def test_serialization_field_order_and_escapes():
    """Field order and escaping must match the model serializer exactly."""
    template = _delta_frame_template(REQUEST_ID, CREATED, MODEL)
    assert template is not None
    for text in ('a"b', "a\\b", "a\nb", "é 日本語 🚀"):
        frame = template_frame(template, 0, text)
        assert frame == model_frame(REQUEST_ID, MODEL, 0, text)
        payload = json.loads(frame[len("data: ") : -2])
        assert list(payload) == ["id", "object", "created", "model", "choices"]
        assert list(payload["choices"][0]) == [
            "index",
            "text",
            "logprobs",
            "finish_reason",
            "matched_stop",
            "token_ids",
            "prompt_token_ids",
        ]


def test_required_tests_are_collected():
    """Fail loudly if the required test names disappear or get skipped."""
    required = {
        "test_template_frame_matches_model_frame",
        "test_first_middle_last_frames_are_byte_identical",
        "test_plain_requests_use_template",
        "test_non_plain_requests_fall_back",
        "test_fallback_frame_keeps_all_fields",
        "test_multi_choice_frames_match_model",
        "test_finish_reason_frame_uses_model_path",
        "test_microbench_template_not_slower_than_model",
        "test_serialization_field_order_and_escapes",
    }
    present = {name for name in globals() if name.startswith("test_")}
    missing = required - present
    assert not missing, f"missing required tests: {sorted(missing)}"


def test_collection_is_nonzero():
    """Guard against 0-collection being mistaken for a pass."""
    collected = [name for name in globals() if name.startswith("test_")]
    assert len(collected) >= 10, f"too few tests collected: {len(collected)}"


def test_usage_and_logprobs_frames_keep_model_fields():
    """Non-plain frames must keep usage/logprobs fields intact."""
    from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor

    usage = UsageProcessor.calculate_token_usage(
        prompt_tokens=3, completion_tokens=2, reasoning_tokens=0
    )
    chunk = CompletionStreamResponse(
        id=REQUEST_ID,
        created=CREATED,
        object="text_completion",
        choices=[
            CompletionResponseStreamChoice(
                index=0,
                text="x",
                logprobs=None,
                finish_reason=None,
                matched_stop=None,
                token_ids=None,
                prompt_token_ids=None,
            )
        ],
        model=MODEL,
        usage=usage,
    )
    payload = json.loads(chunk.model_dump_json())
    assert payload["usage"]["prompt_tokens"] == 3
    assert payload["usage"]["completion_tokens"] == 2
