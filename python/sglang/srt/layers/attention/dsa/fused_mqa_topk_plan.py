from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

# Design port of upstream vLLM #60618 (single-launch fused scoring + top-k for
# the sparse-indexer prefill). The kernel itself is NOT ported here: this module
# only carries over the dispatch contract so that the fused path can be plugged
# in later behind an unchanged caller:
#
#   * eligibility -- one launch covers a chunk of <= 16K query rows and <= 1M
#     keys, multi-request included; topk is one of 512 / 1024 / 2048;
#   * scratch -- a single caller-managed workspace sized once for the maximum
#     shape and reused across layers, so it stays CUDA-graph capture safe;
#   * exact fallback -- the dense path (fp8 MQA logits + top-k) is the only
#     source of truth. A row whose candidate set overflows the selection
#     scratch, or is too small for the gate to bracket the boundary, is
#     rescanned exactly in fp32 instead of returning an approximate top-k.

_ENV_FLAG = "SGLANG_DSA_FUSED_PREFILL_TOPK"

# k values the fused kernel is specialised for.
SUPPORTED_TOPKS: Tuple[int, ...] = (512, 1024, 2048)
# One launch covers at most this many query rows and this many keys.
MAX_ROWS = 16384
MAX_KEYS = 1 << 20
# Below this many visible keys the dense path is cheaper than the gate.
MIN_KEYS = 2 * SUPPORTED_TOPKS[-1]
# Candidate codes the selection scratch holds per row (~11.5K in upstream).
CANDIDATES_PER_ROW = 11776
# Fixed workspace, reused across layers (296 MiB in upstream).
WORKSPACE_BYTES = 296 * 1024**2
# Budget the dense path keeps for one [rows, keys] fp32 logits tile.
DENSE_LOGITS_BUDGET_BYTES = 512 * 1024**2
# fp32 logits are scored into 24-bit codes on the fused path.
SCORE_CODE_BYTES = 4


def _kernel_present() -> bool:
    """Whether the fused kernel binding is importable in this build."""
    try:
        from sglang.kernels.ops.attention.dsa import fused_mqa_topk_prefill  # noqa: F401
    except ImportError:
        return False
    return True


def fused_mqa_topk_available(
    kernel_present: Optional[Callable[[], bool]] = None,
) -> bool:
    """Whether the single-launch fused kernel may be used at all.

    The kernel is not ported yet, so this needs both the binding to exist and an
    explicit opt-in flag; otherwise the caller keeps the dense path.
    """
    if not (kernel_present or _kernel_present)():
        return False
    return os.environ.get(_ENV_FLAG, "0").strip().lower() in ("1", "true", "on")


def fused_mqa_topk_min_keys() -> int:
    """Minimum visible-key count for which the fused path is worth a launch."""
    return MIN_KEYS


@dataclass(frozen=True)
class FusedMQATopKPlan:
    """A prefill chunk that one fused launch can serve."""

    rows: int
    keys: int
    topk: int
    workspace_bytes: int
    dense_chunks: int

    @property
    def dense_logits_bytes(self) -> int:
        """Bytes of the [rows, keys] logits the fused path never materializes."""
        return self.rows * self.keys * SCORE_CODE_BYTES


def fusion_eligible(rows: int, keys: int, topk: int) -> Tuple[bool, str]:
    """Shape check for a single fused launch, with a human-readable reason."""
    if topk not in SUPPORTED_TOPKS:
        return False, f"topk={topk} not in {SUPPORTED_TOPKS}"
    if rows <= 0:
        return False, f"rows={rows} must be positive"
    if rows > MAX_ROWS:
        return False, f"rows={rows} exceeds the single-launch limit {MAX_ROWS}"
    if keys < MIN_KEYS:
        return False, f"keys={keys} below the fusion threshold {MIN_KEYS}"
    if keys > MAX_KEYS:
        return False, f"keys={keys} exceeds the single-launch limit {MAX_KEYS}"
    return True, "eligible"


def dense_logits_chunks(
    rows: int,
    keys: int,
    budget_bytes: int = DENSE_LOGITS_BUDGET_BYTES,
) -> int:
    """Number of dense (logits + top-k) call pairs under the logits budget."""
    if rows <= 0 or keys <= 0:
        return 0
    per_chunk = max(1, budget_bytes // (SCORE_CODE_BYTES * keys))
    return -(-rows // per_chunk)


def row_rescan_required(
    num_candidates: int,
    topk: int,
    capacity: int = CANDIDATES_PER_ROW,
) -> bool:
    """Whether a row must be rescanned exactly instead of selected approximately.

    Mirrors the kernel's recovery rule: too few candidates (the gate failed to
    bracket the top-k boundary) or more than the selection scratch can hold both
    fall back to the exact dense rescan rather than risk a wrong selection.
    """
    return num_candidates < topk or num_candidates > capacity


def resolve_prefill_plan(
    rows: int,
    keys: int,
    topk: int,
    available: Optional[bool] = None,
) -> Tuple[Optional[FusedMQATopKPlan], str]:
    """Plan the fused launch, or return ``None`` and why the dense path stays.

    The dense path stays the only exact reference: a rejected chunk keeps its
    budgeted ``fp8 MQA logits + top-k`` pairs and is never served by an
    approximate fused result.
    """
    if available is None:
        available = fused_mqa_topk_available()
    if not available:
        return None, "fused kernel unavailable or disabled"
    ok, reason = fusion_eligible(rows, keys, topk)
    if not ok:
        return None, reason
    return FusedMQATopKPlan(
        rows=rows,
        keys=keys,
        topk=topk,
        workspace_bytes=WORKSPACE_BYTES,
        dense_chunks=dense_logits_chunks(rows, keys),
    ), reason


def plan_fused_mqa_topk_prefill(
    rows: int,
    keys: int,
    topk: int,
    available: Optional[bool] = None,
) -> Optional[FusedMQATopKPlan]:
    """Plan one fused prefill launch, or ``None`` to keep the dense path."""
    plan, _ = resolve_prefill_plan(rows, keys, topk, available)
    return plan
