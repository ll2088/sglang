# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

"""Platform probing and compatibility checks for the MoE all2all backend.

This module owns the "auto" resolution policy for ``--moe-a2a-backend``:

* :func:`is_gb200_gb300_or_vr200` detects an ARM NVL rack (GB200/GB300/VR200),
  where NVLink spans hosts over MNNVL.
* :func:`is_hgx_b200_or_b300` detects an x86 HGX B200/B300 host, where the
  NVLink domain ends at the node boundary.
* :func:`resolve_moe_a2a_backend` turns the configured value into a concrete
  backend, never returning ``"auto"``.
* :func:`one_sided_unsupported_reason` explains why the one-sided backend was
  not selected, so the fallback is never silent.

The probes are cached pure functions so tests can clear them and simulate
arbitrary hardware without touching the real device.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Backends that are accepted by the CLI but are no longer implemented.
_REMOVED_BACKENDS = ("pplx", "naive")

# The one-sided manager is only wired for these top_k values.
_ONE_SIDED_SUPPORTED_TOP_K = frozenset({1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 22})

# Compute capabilities of the parts the one-sided backend was measured on.
_BLACKWELL_FAMILY = 100  # 10.x, covers 10.0 / 10.3 / 10.7
_HGX_CAPABILITIES = (100, 103)  # HGX B200 (10.0) and HGX B300 (10.3)


def _is_cuda() -> bool:
    try:
        from sglang.srt.utils import is_cuda

        return is_cuda()
    except Exception:
        return False


def _cpu_arch() -> Optional[str]:
    """Return the host CPU architecture as a lowercase string, or None."""
    try:
        import platform

        return platform.machine().lower()
    except Exception:
        return None


def _device_capability() -> Optional[tuple]:
    """Return ``(major, minor)`` of the current device, or None."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return tuple(torch.cuda.get_device_capability())
    except Exception:
        return None


@lru_cache(maxsize=None)
def is_gb200_gb300_or_vr200() -> bool:
    """Whether this is an ARM NVL rack: GB200/GB300 (10.0, 10.3) or VR200 (10.7).

    All of them are Blackwell/Rubin 10.x parts on a Grace or Vera host, where
    NVLink spans hosts over MNNVL.
    """
    if not _is_cuda():
        return False
    arch = _cpu_arch()
    if arch not in ("aarch64", "arm64"):
        # Corner case to distinguish x86 HGX B200/B300 from ARM GB200/GB300/VR200.
        return False
    capability = _device_capability()
    if capability is None:
        return False
    return capability[0] == _BLACKWELL_FAMILY


@lru_cache(maxsize=None)
def is_hgx_b200_or_b300() -> bool:
    """Whether this is an x86 HGX B200 (10.0) or HGX B300 (10.3)."""
    if not _is_cuda():
        return False
    arch = _cpu_arch()
    if arch not in ("x86_64", "amd64"):
        return False
    capability = _device_capability()
    if capability is None:
        return False
    return capability in _HGX_CAPABILITIES


def clear_platform_probes() -> None:
    """Drop the cached platform probes. Intended for tests."""
    is_gb200_gb300_or_vr200.cache_clear()
    is_hgx_b200_or_b300.cache_clear()


def _has_heterogeneous_moe_layers(model_config) -> bool:
    """Whether MoE layers differ in num_experts or top_k (per-layer block_configs)."""
    hf_text_config = getattr(model_config, "hf_text_config", None)
    block_configs = getattr(hf_text_config, "block_configs", None)
    if not block_configs:
        return False

    def field(block, name):
        if isinstance(block, dict):
            return block.get(name)
        return getattr(block, name, None)

    shapes = {
        (field(block, "n_routed_experts"), field(block, "num_experts_per_tok"))
        for block in block_configs
        if field(block, "block_type") == "moe"
    }
    return len(shapes) > 1


def _one_sided_model_unsupported_reason(model_config) -> Optional[str]:
    """Why a model's MoE layers cannot use flashinfer_nvlink_one_sided."""
    if model_config is None:
        return None
    # One shared communication object serves every MoE layer, so all of them
    # must have the same num_experts and top_k.
    if _has_heterogeneous_moe_layers(model_config):
        return "MoE layers with different num_experts or top_k"
    try:
        top_k = model_config.get_num_experts_per_tok()
    except Exception:
        top_k = None
    if top_k and top_k not in _ONE_SIDED_SUPPORTED_TOP_K:
        return f"top_k={top_k}"
    return None


def one_sided_unsupported_reason(
    *,
    enable_expert_parallel: bool,
    enable_elastic_ep: bool,
    nnodes: int,
    data_parallel_size: int,
    data_parallel_size_local: int,
    distributed_executor_backend,
    model_config=None,
    lora_enabled: bool = False,
    moe_backend: str = "auto",
    batch_invariant: bool = False,
) -> Optional[str]:
    """Why the deployment cannot use flashinfer_nvlink_one_sided, if it can't.

    Returns ``None`` when the one-sided backend is supported.
    """
    if not enable_expert_parallel:
        return "expert parallelism is disabled"
    # The one-sided manager is not wired for elastic EP's stateless groups.
    if enable_elastic_ep:
        return "elastic EP"
    if lora_enabled:
        return "LoRA"
    # Experts picked with --moe-runner-backend decide on their own whether they
    # take one-sided dispatch, so any explicit choice keeps the old default.
    if moe_backend != "auto":
        return f"moe_backend={moe_backend!r}"
    # No kernel is both batch invariant and one-sided compatible.
    if batch_invariant:
        return "VLLM_BATCH_INVARIANT=1"
    if reason := _one_sided_model_unsupported_reason(model_config):
        return reason
    # GB200/GB300/VR200 carry NVLink between hosts over MNNVL, so the EP group
    # may span nodes and any data-parallel size is served.
    if is_gb200_gb300_or_vr200():
        return None
    # HGX has no NVLink between hosts, so the whole deployment must sit on one
    # physical node, placed by SGLang's own launcher or by Ray. Parallelism
    # layout (DP, TP, EP size) does not matter. An API-server-only process has
    # no local ranks and must agree with engines that do.
    if nnodes != 1:
        return f"nnodes={nnodes} on a GPU without NVLink between hosts"
    if data_parallel_size_local not in (0, data_parallel_size):
        return (
            "data_parallel_size_local="
            f"{data_parallel_size_local} does not cover data_parallel_size="
            f"{data_parallel_size}"
        )
    if distributed_executor_backend not in ("mp", "uni", "ray"):
        return f"distributed_executor_backend={distributed_executor_backend!r}"
    if not is_hgx_b200_or_b300():
        return "GPU is not a Blackwell/Rubin part with measured one-sided gains"
    return None


def resolve_moe_a2a_backend(
    requested: str,
    *,
    enable_expert_parallel: bool,
    enable_elastic_ep: bool,
    nnodes: int,
    data_parallel_size: int,
    data_parallel_size_local: int,
    distributed_executor_backend,
    model_config=None,
    lora_enabled: bool = False,
    moe_backend: str = "auto",
    batch_invariant: bool = False,
) -> tuple:
    """Resolve the configured all2all backend into a concrete one.

    Returns:
        ``(backend, auto_one_sided, unsupported_reason)``. ``backend`` is never
        ``"auto"``; ``auto_one_sided`` records whether the auto policy picked
        the one-sided backend, so the manager can produce a clearer error when
        it later detects a multi-node EP group.
    """
    if requested in _REMOVED_BACKENDS:
        logger.warning(
            "The '%s' all2all backend has been removed. "
            "Falling back to 'none'.",
            requested,
        )
        return "none", False, f"backend {requested!r} was removed"
    if requested != "auto":
        return requested, False, None
    reason = one_sided_unsupported_reason(
        enable_expert_parallel=enable_expert_parallel,
        enable_elastic_ep=enable_elastic_ep,
        nnodes=nnodes,
        data_parallel_size=data_parallel_size,
        data_parallel_size_local=data_parallel_size_local,
        distributed_executor_backend=distributed_executor_backend,
        model_config=model_config,
        lora_enabled=lora_enabled,
        moe_backend=moe_backend,
        batch_invariant=batch_invariant,
    )
    if reason is None:
        return "flashinfer_nvlink_one_sided", True, None
    logger.info(
        "all2all backend 'auto' resolved to 'none': %s. "
        "Pass --moe-a2a-backend explicitly to override.",
        reason,
    )
    return "none", False, reason


def require_single_node_without_mnnvl(auto_one_sided: bool = False) -> None:
    """Fail clearly if one-sided all2all would span hosts without MNNVL.

    GB200/GB300/VR200 connect hosts over MNNVL. On other hardware (HGX B200 and
    B300) the NVLink domain ends at the node, so every rank must share one host.
    """
    if is_gb200_gb300_or_vr200():
        return
    try:
        from sglang.srt.distributed.parallel_state import get_node_count

        node_count = get_node_count()
    except Exception:
        # Used outside SGLang's distributed environment: node count is unknown.
        return
    if node_count == 1:
        return
    origin = (
        " It was picked by default (--moe-a2a-backend auto), which assumes a "
        "single physical node."
        if auto_one_sided
        else ""
    )
    raise RuntimeError(
        "The 'flashinfer_nvlink_one_sided' all2all backend needs every rank on "
        "one physical node on this GPU, which has no NVLink between hosts, but "
        f"the ranks span {node_count} nodes.{origin} Switch to the 'none' "
        "backend manually with `--moe-a2a-backend none`."
    )
