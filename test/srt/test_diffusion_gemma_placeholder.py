# Copyright 2026 SGLang Team
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

import re

from sglang.srt.multimodal.processors.diffusion_gemma import (
    DiffusionGemmaSGLangProcessor,
)


class _FakeProcessor:
    """Minimal stand-in for the HF processor used by MultimodalSpecialTokens."""

    def __init__(self):
        self.image_token = "<|image|>"
        self.image_token_id = 0


def test_diffusion_gemma_image_token_matches_replacement_target():
    """The image token written into the prompt must be the one the
    replacement regex targets, mirroring the upstream Gemma 4 fix.
    """
    processor = _FakeProcessor()
    tokens = DiffusionGemmaSGLangProcessor.__dict__["__init__"]
    # Build the same MultimodalSpecialTokens the processor builds, without
    # instantiating the full server-side processor.
    from sglang.srt.multimodal.processors.base_processor import (
        MultimodalSpecialTokens,
    )

    mm_tokens = MultimodalSpecialTokens(
        image_token="<|image|>",
        image_token_id=processor.image_token_id,
        image_token_regex=re.compile(
            r"<\|image>(?:<\|image\|>)+<image\|>|<\|image\|>"
        ),
    ).build(processor)

    assert mm_tokens.image_token == "<|image|>"
    assert mm_tokens.image_token_regex.search("<|image|>") is not None
    assert mm_tokens.image_token_regex.search("<image_soft_token>") is None
