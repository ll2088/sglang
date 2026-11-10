# SPDX-License-Identifier: Apache-2.0
"""Dtype-aware truncated SVD used by SVDQuant-style calibration.

The rank-truncated factors are cast back to the weight dtype, so computing the
SVD in float64 buys nothing unless the weight itself is float64. float64 SVD is
over 20x slower on GPUs with low FP64 throughput (e.g. GB300/B300).
"""

from __future__ import annotations

import torch


def svd_compute_dtype(original_dtype: torch.dtype) -> torch.dtype:
    """Return the SVD compute dtype matching the output dtype."""
    return torch.float64 if original_dtype == torch.float64 else torch.float32


def svd(
    weight: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-truncated SVD returning ``(us, vt)`` in the weight dtype.

    ``us`` is ``u[:, :rank] * s[:rank]`` and ``vt`` is ``vt[:rank]``, both cast
    back to ``weight.dtype`` on ``weight.device``.
    """
    original_device = weight.device
    original_dtype = weight.dtype
    compute_dtype = svd_compute_dtype(original_dtype)
    u, s, vt = torch.linalg.svd(
        weight.to(dtype=compute_dtype, device=original_device),
        full_matrices=False,
    )
    us = u[:, :rank] * s[:rank]
    vt = vt[:rank]
    us = us.to(device=original_device, dtype=original_dtype)
    vt = vt.to(device=original_device, dtype=original_dtype)
    return us, vt
