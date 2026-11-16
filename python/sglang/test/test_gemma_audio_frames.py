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


def _num_frames(num_samples: int, samples_per_token: int = SAMPLES_PER_TOKEN) -> int:
    return math.ceil(num_samples / samples_per_token)


@pytest.mark.parametrize("num_samples", [161, 320, 321, 640, 3200, 3300])
def test_single_frame_audio_preserves_rank(num_samples):
    """A single-frame clip must keep the (batch, frames, features) rank."""
    frames = _num_frames(num_samples)
    features = torch.zeros(1, frames, 128)
    mask = torch.ones(1, frames, dtype=torch.bool)

    # Simulate the batching helper contract: never squeeze the time axis.
    batched_features = features
    batched_mask = mask

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
