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
"""Parity tests for the plain-delta completion SSE frame template.

Frames rendered from the per-stream template must be byte-identical to the
frames the response models would have produced, and requests that can emit
non-plain chunks must keep using the model path.
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import (
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
)
from sglang.srt.entrypoints.openai.sse_utils import (
    build_completion_delta_frame_template,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CREATED = 1706000000


def model_frame(
    chunk_id: str,
    model_name: str,
    index: int,
    text: str,
    finish_reason=None,
    matched_stop=None,
    token_ids=None,
    prompt_token_ids=None,
) -> str:
    """Render a frame through the response models, the reference path."""
    chunk = CompletionStreamResponse(
        id=chunk_id,
        created=CREATED,
        object="text_completion",
        choices=[
            CompletionResponseStreamChoice(
                index=index,
                text=text,
                logprobs=None,
                finish_reason=finish_reason,
                matched_stop=matched_stop,
                token_ids=token_ids,
                prompt_token_ids=prompt_token_ids,
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


class TestCompletionDeltaFrameTemplate(unittest.TestCase):
    """Byte-level parity between the template path and the model path."""

    def _template(self, chunk_id="cmpl-abc123", model_name="test-model"):
        template = build_completion_delta_frame_template(
            chunk_id,
            CREATED,
            model_name,
            CompletionStreamResponse,
            CompletionResponseStreamChoice,
        )
        self.assertIsNotNone(template)
        return template

    def test_parity_plain_delta_texts(self):
        """Template frames match model frames for plain delta text."""
        template = self._template()
        for text in [
            "",
            "Hello, world",
            ' "quoted" and \\backslashed\\ ',
            "line\nbreaks\r\ttabs",
            "\x00\x01\x1f\x7f",
            "unicode \u00e9 \u65e5\u672c\u8a9e \U0001f680 \U0001d549\U0010ffff",
        ]:
            with self.subTest(text=text):
                self.assertEqual(
                    template_frame(template, 0, text),
                    model_frame("cmpl-abc123", "test-model", 0, text),
                )

    def test_parity_multiple_indices_and_ids(self):
        """Parity holds across choice indices and request/model strings."""
        cases = [
            ("cmpl-abc123", "meta-llama/Llama-3.1-8B", 0),
            ('id "with" quotes\\', "\u043c\u043e\u0434\u0435\u043b\u044c \U0001f3af", 7),
        ]
        for chunk_id, model_name, index in cases:
            template = self._template(chunk_id, model_name)
            for text in ["", "hi", "\u00e9\n"]:
                with self.subTest(chunk_id=chunk_id, index=index, text=text):
                    self.assertEqual(
                        template_frame(template, index, text),
                        model_frame(chunk_id, model_name, index, text),
                    )

    def test_parity_first_and_last_frame_boundaries(self):
        """First (empty) and last (finish_reason) frames behave as expected."""
        template = self._template()
        # First frame: empty delta still renders through the template.
        self.assertEqual(
            template_frame(template, 0, ""),
            model_frame("cmpl-abc123", "test-model", 0, ""),
        )
        # Terminal frame carries finish_reason and must use the model path.
        terminal = model_frame(
            "cmpl-abc123", "test-model", 0, " end", finish_reason="stop"
        )
        self.assertIn('"finish_reason":"stop"', terminal)
        self.assertNotEqual(template_frame(template, 0, " end"), terminal)

    def test_template_is_none_when_sentinels_are_ambiguous(self):
        """Ambiguous sentinels must disable the template path."""
        self.assertIsNone(
            build_completion_delta_frame_template(
                "cmpl-987654321",
                CREATED,
                "m",
                CompletionStreamResponse,
                CompletionResponseStreamChoice,
            )
        )

    def test_template_constant_fields_match_probe(self):
        """Template constants equal the per-stream constant fields."""
        template = self._template()
        prefix, infix, suffix = template
        frame = template_frame(template, 0, "x")
        self.assertTrue(frame.startswith("data: "))
        self.assertTrue(frame.endswith("\n\n"))
        self.assertIn('"object":"text_completion"', frame)
        self.assertIn('"model":"test-model"', frame)
        self.assertIn(str(CREATED), frame)
        self.assertIn('"id":"cmpl-abc123"', frame)
        self.assertEqual(prefix + "0" + infix + '"x"' + suffix, frame)


class TestCompletionStreamFallback(unittest.TestCase):
    """Non-plain requests must keep the model path output unchanged."""

    def test_non_plain_fields_use_model_path(self):
        """Frames with token ids or finish_reason match model serialization."""
        cases = [
            dict(finish_reason="stop"),
            dict(matched_stop=" end"),
            dict(token_ids=[1, 2, 3]),
            dict(prompt_token_ids=[4, 5]),
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                frame = model_frame(
                    "cmpl-abc123", "test-model", 0, "hi", **kwargs
                )
                data = json.loads(frame[len("data: ") : -2])
                choice = data["choices"][0]
                for key, value in kwargs.items():
                    self.assertEqual(choice[key], value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
