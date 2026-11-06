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
from typing import Dict, List, Optional, Union

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.gemma4_diffusion import DiffusionGemmaForBlockDiffusion
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens

# Placeholder written into the prompt when the frontend renders multimodal
# content (e.g. `--chat-template-content-format string`). It must be exactly
# the token targeted by `IMAGE_TOKEN_REGEX` below, otherwise the prompt
# replacement fails to find it in the prompt text.
IMAGE_PLACEHOLDER = "<|image|>"


def _build_image_token_regex() -> "re.Pattern[str]":
    """Build the regex matching the raw and the HF-expanded image placeholder.

    The HF expansion is `boi + soft tokens + eoi`, i.e. `<|image>` followed by
    one or more placeholders and `<image|>`.
    """
    pattern = re.compile(
        r"<\|image>(?:" + re.escape(IMAGE_PLACEHOLDER) + r")+<image\|>"
        r"|" + re.escape(IMAGE_PLACEHOLDER)
    )
    if pattern.fullmatch(IMAGE_PLACEHOLDER) is None:
        raise ValueError(
            f"Image placeholder {IMAGE_PLACEHOLDER!r} is not targeted by the "
            "prompt replacement regex"
        )
    return pattern


IMAGE_TOKEN_REGEX = _build_image_token_regex()


class DiffusionGemmaSGLangProcessor(SGLangBaseProcessor):
    """DiffusionGemma image processor."""

    models = [DiffusionGemmaForBlockDiffusion]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=IMAGE_PLACEHOLDER,
            image_token_id=hf_config.image_token_id,
            # Also match the HF-expanded placeholder (boi + soft tokens + eoi).
            image_token_regex=IMAGE_TOKEN_REGEX,
        ).build(_processor)
        self.ATTR_NAME_TO_MODALITY["image_position_ids"] = Modality.IMAGE

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj=None,
        *args,
        **kwargs,
    ):
        if kwargs.get("audio_data") or getattr(request_obj, "video_data", None):
            raise ValueError(
                "DiffusionGemma serving supports image input only "
                "(audio/video are not implemented)."
            )
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )
