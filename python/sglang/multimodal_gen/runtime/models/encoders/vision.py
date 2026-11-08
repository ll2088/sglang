# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/vision.py

from typing import Any, NamedTuple

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_VISION_LUT_VALUES = 256


class VisionNormalization(NamedTuple):
    """Elementwise normalization a HF image processor applies to uint8 pixels."""

    rescale_factor: float | None
    image_mean: tuple[float, ...] | None
    image_std: tuple[float, ...] | None
    out_dtype: torch.dtype


def _as_channel_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return (float(value),)
    try:
        channels = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return channels or None


def vision_normalization_from_processor(
    processor: Any, out_dtype: torch.dtype
) -> VisionNormalization | None:
    """Describe the per-channel normalization `processor` applies on the host.

    Returns ``None`` when the settings cannot be expressed as one affine
    transform per channel, in which case callers must keep the CPU path.
    """
    if processor is None:
        return None
    do_rescale = bool(getattr(processor, "do_rescale", False))
    do_normalize = bool(getattr(processor, "do_normalize", False))
    if not (do_rescale or do_normalize):
        return None
    rescale_factor = getattr(processor, "rescale_factor", None)
    if do_rescale and rescale_factor is None:
        return None
    image_mean = image_std = None
    if do_normalize:
        image_mean = _as_channel_tuple(getattr(processor, "image_mean", None))
        image_std = _as_channel_tuple(getattr(processor, "image_std", None))
        if image_mean is None or image_std is None:
            return None
        if len(image_mean) != len(image_std):
            return None
    return VisionNormalization(
        rescale_factor=float(rescale_factor) if do_rescale else None,
        image_mean=image_mean,
        image_std=image_std,
        out_dtype=out_dtype,
    )


def build_vision_pixel_lookup_table(
    normalization: VisionNormalization,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Precompute the host normalization output for every uint8 value.

    The table is built with the same float32 arithmetic the host path uses, so
    the device result is bitwise identical to normalizing on the CPU.
    """
    channels = 1 if normalization.image_mean is None else len(normalization.image_mean)
    values = torch.arange(_VISION_LUT_VALUES, dtype=torch.float32).expand(channels, -1)
    if normalization.rescale_factor is not None:
        values = values * normalization.rescale_factor
    if normalization.image_mean is not None:
        mean = torch.tensor(normalization.image_mean, dtype=torch.float32)[:, None]
        std = torch.tensor(normalization.image_std, dtype=torch.float32)[:, None]
        values = (values - mean) / std
    return values.to(device=device, dtype=normalization.out_dtype)


def normalize_vision_pixels(
    pixels: torch.Tensor,
    normalization: VisionNormalization | None,
    table: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize uint8 pixels on device with the per-channel lookup table.

    The channel index is ``position % channels`` along the last dimension, which
    covers NHWC images and the flattened ``C * T * P * P`` patch layout vision
    encoders consume. Non-uint8 pixels (already normalized on the host) and
    shapes the table does not cover are returned unchanged: that is the CPU
    fallback and keeps this helper free of value guessing.
    """
    if normalization is None or pixels.dtype != torch.uint8 or pixels.dim() == 0:
        return pixels
    if table is None:
        table = build_vision_pixel_lookup_table(normalization, pixels.device)
    channels = table.shape[0]
    width = pixels.shape[-1]
    if channels == 0 or width % channels != 0:
        return pixels
    offsets = torch.arange(width, device=pixels.device, dtype=torch.int64) % channels
    flat_index = pixels.to(torch.int64) * channels + offsets
    return table.reshape(-1).take(flat_index)


def resolve_visual_encoder_outputs(
    encoder_outputs: torch.Tensor | list[torch.Tensor],
    feature_sample_layers: list[int] | None,
    post_layer_norm: torch.nn.LayerNorm | None,
    max_possible_layers: int,
) -> torch.Tensor:
    """Given the outputs a visual encoder module that may correspond to the
    output of the last layer, or a list of hidden states to be stacked,
    handle post normalization and resolve it into a single output tensor.

    Args:
        encoder_outputs: Output of encoder's last layer or all hidden states.
        feature_sample_layers: Optional layer indices to grab from the encoder
            outputs; if provided, encoder outputs must be a list.
        post_layer_norm: Post norm to apply to the output of the encoder.
        max_possible_layers: Total layers in the fully loaded visual encoder.

    """
    if feature_sample_layers is None:
        if post_layer_norm is not None:
            return post_layer_norm(encoder_outputs)
        return encoder_outputs

    # Get the hidden states corresponding to the layer indices.
    # Negative values are relative to the full visual encoder,
    # so offset them depending on how many layers were loaded.
    # NOTE: this assumes that encoder_outputs is a list containing
    # the inputs to the visual encoder, followed by the hidden states
    # of each layer.
    num_loaded_layers = len(encoder_outputs) - 1
    offset = max_possible_layers - num_loaded_layers
    hs_pool = [
        (
            encoder_outputs[layer_idx]
            if layer_idx >= 0
            else encoder_outputs[layer_idx + offset]
        )
        for layer_idx in feature_sample_layers
    ]

    # Apply post-norm on the final hidden state if we are using it
    uses_last_layer = feature_sample_layers[-1] in (len(hs_pool) - 1, -1)
    if post_layer_norm is not None and uses_last_layer:
        hs_pool[-1] = post_layer_norm(encoder_outputs)
    return torch.cat(hs_pool, dim=-1)
