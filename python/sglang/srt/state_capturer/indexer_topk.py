import logging
from typing import Optional

import numpy as np

from sglang.srt.configs.model_config import ModelConfig, get_num_indexer_layers
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_resources,
    get_schedule,
)
from sglang.srt.state_capturer.base import BaseTopkCapturer

logger = logging.getLogger(__name__)


def _lazy_import_torch():
    """Import torch only when a capturer is actually constructed.

    Keeping the module importable without torch lets the backend-resolution and
    fallback helpers be unit-tested in a minimal numpy/pytest environment.
    """
    import torch  # noqa: PLC0415

    return torch


# BF16 logits are read natively only when the row stride keeps every row
# 16-byte aligned (8 BF16 elements).  Mirrors the kernel-side predicate in
# kpool_topk_transform.cuh so the dispatcher and the kernel agree.
_BF16_VEC_WIDTH = 8


def takes_bf16_logits_for(
    dtype_code: int,
    dtype_bits: int,
    dtype_lanes: int,
    row_stride: int,
) -> bool:
    """Host-side mirror of the kernel's BF16 native-read predicate.

    Kept free of tensor types so the dtype/alignment negotiation can be
    unit-tested without CUDA or torch.  ``dtype_code`` follows DLPack
    (kDLBfloat == 4).
    """
    return (
        dtype_code == 4
        and dtype_bits == 16
        and dtype_lanes == 1
        and row_stride % _BF16_VEC_WIDTH == 0
    )


def resolve_topk_logits_dtype(
    dtype_code: int,
    dtype_bits: int,
    dtype_lanes: int,
    row_stride: int,
    backend: str = "cooperative",
) -> str:
    """Negotiate the logits dtype a top-k backend should read.

    Returns ``"bf16"`` when the backend can read BF16 natively and the row
    stride keeps every row 16-byte aligned, otherwise ``"fp32"``.  The caller
    upcasts to FP32 on the fallback path, so the selection is unchanged.
    """
    if backend not in ("cooperative", "persistent"):
        return "fp32"
    if takes_bf16_logits_for(dtype_code, dtype_bits, dtype_lanes, row_stride):
        return "bf16"
    return "fp32"


def resolve_topk_logits_dtype_or_fallback(
    dtype_code: int,
    dtype_bits: int,
    dtype_lanes: int,
    row_stride: int,
    backend: str = "cooperative",
) -> str:
    """Same negotiation as :func:`resolve_topk_logits_dtype`, but never raises.

    Kept as a separate entry point so callers that must not fail on an
    unexpected dtype (e.g. a backend that only takes FP32) can fall back to
    ``"fp32"`` and record the reason.  Pure Python, no torch/CUDA needed.
    """
    try:
        return resolve_topk_logits_dtype(
            dtype_code, dtype_bits, dtype_lanes, row_stride, backend
        )
    except Exception:  # noqa: BLE001 - fallback must never propagate
        return "fp32"


class IndexerTopkCapturer(BaseTopkCapturer):
    def __init__(
        self,
        num_tokens: int,
        num_indexer_layers: int,
        index_topk: int,
        max_running_requests: int,
        device: str,
    ):
        _lazy_import_torch()
        self.num_indexer_layers = num_indexer_layers
        self.index_topk = index_topk

        attn_tp_size = get_parallel().attn_tp_size
        assert attn_tp_size == 1, "IndexerTopkCapturer now only supports DP attention"

        # DP-attention capture is per-rank-local: each rank writes [:local_batch, ...]
        # to its own device_cache, so the buffer only needs to fit one rank's batch.
        max_batch_size = max(get_schedule().chunked_prefill_size, max_running_requests)

        super().__init__(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=self.num_indexer_layers,
            topk_size=self.index_topk,
            device=device,
            name="indexer_topk",
        )


def get_global_indexer_capturer() -> Optional[IndexerTopkCapturer]:

    return get_resources().indexer_capturer


def set_global_indexer_capturer(capturer: Optional[IndexerTopkCapturer]):

    get_resources().indexer_capturer = capturer


def destroy_global_indexer_capturer():
    if (capturer := get_resources().indexer_capturer) is not None:
        capturer.destroy()
    get_resources().indexer_capturer = None


def maybe_capture_indexer_topk(
    layer_id: int, topk_indices: Optional["torch.Tensor"]
) -> Optional["torch.Tensor"]:
    """Capture topk for layer_id if a capturer is set; pass through unchanged.

    Works in both expression context (`return maybe_capture_indexer_topk(...)`)
    and statement context (call for side-effect, ignore return).
    """
    if topk_indices is None:
        return None
    if (cap := get_global_indexer_capturer()) is not None:
        cap.capture(layer_id=layer_id, topk_indices=topk_indices)
    return topk_indices


def extract_indexer_topk_from_meta_info(data):
    # Mirrors extract_routed_experts_from_meta_info: indices are returned as
    # base64-encoded int32 bytes. Caller reshapes to (seqlen-1, num_indexer_layers,
    # index_topk).
    import pybase64  # noqa: PLC0415

    indexer_topk_base64 = data["meta_info"].get("indexer_topk", None)
    if indexer_topk_base64 is None:
        return np.empty((0,), dtype=np.int32)
    indexer_topk = np.frombuffer(
        pybase64.b64decode(indexer_topk_base64.encode("utf-8")), dtype=np.int32
    )
    return indexer_topk


def create_indexer_capturer(
    *,
    model_config: ModelConfig,
    num_tokens: int,
    max_running_requests: int,
    device: str,
) -> Optional[IndexerTopkCapturer]:

    enable = get_exec().features.enable_return_indexer_topk
    # Producer wiring is CUDA/NPU-only; other backends
    # would create a capturer but never feed it.
    if enable and device != "cuda" and device != "npu":
        logger.warning(
            "indexer-topk capture is not wired for %s backend. Disabling capturer.",
            device,
        )
        return None

    hf_text_config = model_config.hf_text_config
    num_indexer_layers = get_num_indexer_layers(hf_text_config)
    index_topk = getattr(hf_text_config, "index_topk", 0)
    return _create_indexer_capturer_raw(
        enable=enable,
        num_indexer_layers=num_indexer_layers,
        index_topk=index_topk,
        num_tokens=num_tokens,
        max_running_requests=max_running_requests,
        device=device,
    )


def _create_indexer_capturer_raw(
    enable: bool,
    num_indexer_layers: int,
    index_topk: int,
    num_tokens: int,
    max_running_requests: int,
    device: str,
) -> Optional[IndexerTopkCapturer]:
    if not enable:
        return None
    if num_indexer_layers <= 0:
        logger.warning("No indexer layers found, IndexerTopkCapturer disabled")
        return None
    if index_topk <= 0:
        logger.warning(
            "Invalid index_topk=%s, IndexerTopkCapturer disabled", index_topk
        )
        return None
    return IndexerTopkCapturer(
        num_tokens=num_tokens,
        num_indexer_layers=num_indexer_layers,
        index_topk=index_topk,
        max_running_requests=max_running_requests,
        device=device,
    )
