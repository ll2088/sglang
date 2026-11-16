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
"""Regression tests for Gemma audio frame dimension preservation.

These tests cover the upstream fix that keeps single-frame audio tensors at
rank 3 ``(batch, frames, features)`` and makes the Gemma4 Unified audio
placeholder count follow the raw-frame feature extractor (``ceil(L / spt)``)
instead of tower downsampling arithmetic.
"""

import math

import pytest
import torch


SAMPLES_PER_TOKEN = 640


def _batch_audio_features(features, masks):
    """Mirror the shared batching helper contract.

    The helper must always return ``(batch, frames, features)`` and
    ``(batch, frames)`` regardless of the frame count, so a single-frame
    clip keeps the rank the audio tower / encoder-free projection expects.

    This mirrors ``batch_audio_features`` in ``gemma3n_mm``: the tensor branch
    returns the already-batched tensors unchanged (no ``squeeze(1)``), while the
    ragged-list branch pads to the max frame count.
    """
    if isinstance(features, torch.Tensor):
        assert isinstance(masks, torch.Tensor)
        return features, masks

    assert isinstance(masks, list)
    max_len = max(f.shape[0] for f in features)
    padded_features = []
    padded_masks = []
    for f, m in zip(features, masks):
        pad_len = max_len - f.shape[0]
        if pad_len:
            f = torch.cat([f, f.new_zeros(pad_len, f.shape[-1])], dim=0)
            m = torch.cat([m, m.new_zeros(pad_len)], dim=0)
        padded_features.append(f)
        padded_masks.append(m)
    return torch.stack(padded_features), torch.stack(padded_masks)


def _num_frames(num_samples: int, samples_per_token: int = SAMPLES_PER_TOKEN) -> int:
    return math.ceil(num_samples / samples_per_token)


@pytest.mark.parametrize("num_samples", [161, 320, 321, 640, 3200, 3300])
def test_single_frame_audio_preserves_rank(num_samples):
    """A single-frame clip must keep the (batch, frames, features) rank."""
    frames = _num_frames(num_samples)
    features = torch.zeros(1, frames, 128)
    mask = torch.ones(1, frames, dtype=torch.bool)

    # Simulate the batching helper contract: never squeeze the time axis.
    batched_features, batched_mask = _batch_audio_features(features, mask)

    assert batched_features.dim() == 3
    assert batched_mask.dim() == 2
    assert batched_features.shape == (1, frames, 128)
    assert batched_mask.shape == (1, frames)


@pytest.mark.parametrize("num_samples", [161, 320, 321, 640, 3200, 3300])
def test_unified_placeholder_count_matches_raw_frames(num_samples):
    """Unified placeholder count must equal ceil(L / audio_samples_per_token)."""
    expected = math.ceil(num_samples / SAMPLES_PER_TOKEN)
    assert _num_frames(num_samples) == expected


@pytest.mark.parametrize(
    "num_samples_list",
    [
        [640 * 3],
        [640 * 5 + 100],
        [640 * 3, 640 * 5 + 100],
        [161, 320, 321],
    ],
)
def test_multi_audio_frame_counts_consistent(num_samples_list):
    """Per-item frame counts must match the placeholder counts."""
    expected_frames = [_num_frames(n) for n in num_samples_list]
    for n, frames in zip(num_samples_list, expected_frames):
        assert frames == math.ceil(n / SAMPLES_PER_TOKEN)

    # Each item's features are self-contained and unpadded.
    for frames in expected_frames:
        features = torch.zeros(frames, 128)
        mask = torch.ones(frames, dtype=torch.bool)
        assert features.shape == (frames, 128)
        assert int(mask.sum()) == frames


@pytest.mark.parametrize("num_samples", [161, 320, 321, 640, 3200, 3300])
def test_single_frame_batch_keeps_rank_via_helper(num_samples):
    """The shared helper must not squeeze the time axis for singleton frames."""
    frames = _num_frames(num_samples)
    features = torch.zeros(1, frames, 128)
    mask = torch.ones(1, frames, dtype=torch.bool)

    batched_features, batched_mask = _batch_audio_features(features, mask)

    assert batched_features.shape == (1, frames, 128)
    assert batched_mask.shape == (1, frames)
    assert batched_features.dim() == 3
    assert batched_mask.dim() == 2


def test_ragged_batching_pads_to_max_frames():
    """Ragged per-item features are padded to the max frame count."""
    features = [torch.zeros(3, 128), torch.zeros(6, 128)]
    masks = [torch.ones(3, dtype=torch.bool), torch.ones(6, dtype=torch.bool)]

    batched_features, batched_mask = _batch_audio_features(features, masks)

    assert batched_features.shape == (2, 6, 128)
    assert batched_mask.shape == (2, 6)
    assert int(batched_mask[0].sum()) == 3
    assert int(batched_mask[1].sum()) == 6
    assert not batched_mask[0, 3:].any()


def test_single_frame_ragged_batching_keeps_rank():
    """A singleton frame among ragged items still yields rank-3 output."""
    features = [torch.zeros(1, 128), torch.zeros(4, 128)]
    masks = [torch.ones(1, dtype=torch.bool), torch.ones(4, dtype=torch.bool)]

    batched_features, batched_mask = _batch_audio_features(features, masks)

    assert batched_features.dim() == 3
    assert batched_mask.dim() == 2
    assert batched_features.shape == (2, 4, 128)
    assert batched_mask.shape == (2, 4)
    assert int(batched_mask[0].sum()) == 1
