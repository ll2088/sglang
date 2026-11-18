# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

"""Platform-aware resolution of the MoE all-to-all backend.

This module mirrors the strategy used upstream by vLLM: the default backend
is no longer a static constant but is resolved against the detected hardware
and deployment topology.  Only the combinations that were measured to be a
win are switched to ``flashinfer_nvlink_one_sided``; everything else keeps the
previous default so that behaviour is unchanged on unsupported platforms.
"""

from __future__ import annotations

import logging
from functools import cache
from typing import Optional

import torch

from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

# Backends that are accepted by ``--moe-a2a-backend``.  ``auto`` is a sentinel
# that is resolved by :func:`resolve_moe_a2a_backend` before use.
MOE_A2A_BACKENDS = (
    "auto",
    "none",
    "deepep",
    "mooncake",
    "nixl",
    "mori",
    "flashinfer",
    "flashinfer_nvlink_one_sided",
    "ascend_fuseep",
    "pplx",
)

# The backend used when ``auto`` does not pick the one-sided manager.
DEFAULT_MOE_A2A_BACKEND = "none"

# The backend enabled by ``auto`` on the hardware where it was measured to win.
ONE_SIDED_MOE_A2A_BACKEND = "flashinfer_nvlink_one_sided"


def _device_capability() -> Optional[tuple[int, int]]:
    """Return the compute capability of the current device, or ``None``."""
    try:
        return torch.cuda.get_device_capability()
    except Exception:
        return None


def _cpu_architecture() -> Optional[str]:
    """Return the host CPU architecture as a lowercase string, or ``None``."""
    try:
        import platform

        return platform.machine().lower()
    except Exception:
        return None


@cache
def _is_gb200_gb300_or_vr200() -> bool:
    """Whether this is an ARM NVL rack: GB200/GB300 (10.0/10.3) or VR200 (10.7).

    All of them are Blackwell/Rubin 10.x parts on a Grace or Vera host, where
    NVLink spans hosts over MNNVL so the EP group may cross node boundaries.
    """
    if not is_cuda():
        return False
    arch = _cpu_architecture()
    if arch not in ("aarch64", "arm64"):
        return False
    capability = _device_capability()
    if capability is None:
        return False
    return capability[0] == 10


@cache
def _is_hgx_b200_or_b300() -> bool:
    """Whether this is an x86 HGX B200 (10.0) or HGX B300 (10.3)."""
    if not is_cuda():
        return False
    arch = _cpu_architecture()
    if arch not in ("x86_64", "amd64"):
        return False
    capability = _device_capability()
    if capability is None:
        return False
    return capability in ((10, 0), (10, 3))


def clear_platform_probes() -> None:
    """Clear the cached platform probes (used by tests)."""
    _is_gb200_gb300_or_vr200.cache_clear()
    _is_hgx_b200_or_b300.cache_clear()


def _prefers_one_sided_all2all(
    *,
    enable_expert_parallel: bool,
    enable_elastic_ep: bool,
    nnodes: int,
    data_parallel_size: int,
    data_parallel_size_local: int,
) -> bool:
    """Whether ``auto`` picks ``flashinfer_nvlink_one_sided``.

    Only the hardware combinations where the one-sided manager was measured to
    beat the previous default are enabled.  The one-sided manager is not wired
    for elastic EP's stateless groups, so elastic EP always falls back.
    """
    if not enable_expert_parallel or enable_elastic_ep:
        return False
    # GB200/GB300/VR200 carry NVLink between hosts over MNNVL, so the EP group
    # may span nodes and any data-parallel size is served.
    if _is_gb200_gb300_or_vr200():
        return True
    # HGX has no NVLink between hosts, so the whole deployment must sit on one
    # physical node.  An API-server-only process has no local ranks and must
    # agree with the engines that do.
    return (
        nnodes == 1
        and data_parallel_size_local in (0, data_parallel_size)
        and _is_hgx_b200_or_b300()
    )


def resolve_moe_a2a_backend(
    requested: str,
    *,
    enable_expert_parallel: bool = False,
    enable_elastic_ep: bool = False,
    nnodes: int = 1,
    data_parallel_size: int = 1,
    data_parallel_size_local: int = 0,
) -> str:
    """Resolve the configured MoE all-to-all backend to a concrete value.

    Args:
        requested: The value of ``--moe-a2a-backend``.  ``auto`` is resolved
            against the detected platform and deployment topology; any other
            value is returned unchanged so that explicit user choices win.

    Returns:
        A concrete backend name; never ``auto``.
    """
    if requested != "auto":
        return requested
    if _prefers_one_sided_all2all(
        enable_expert_parallel=enable_expert_parallel,
        enable_elastic_ep=enable_elastic_ep,
        nnodes=nnodes,
        data_parallel_size=data_parallel_size,
        data_parallel_size_local=data_parallel_size_local,
    ):
        logger.info(
            "moe_a2a_backend='auto' resolved to '%s' on this platform.",
            ONE_SIDED_MOE_A2A_BACKEND,
        )
        return ONE_SIDED_MOE_A2A_BACKEND
    return DEFAULT_MOE_A2A_BACKEND


def require_single_node_without_mnnvl(node_count: Optional[int]) -> None:
    """Fail clearly if one-sided all2all would span hosts without MNNVL.

    GB200/GB300/VR200 connect hosts over MNNVL.  On other hardware (HGX B200
    and B300) the NVLink domain ends at the node, so every rank must share one
    host.  ``node_count`` is ``None`` when the node count is unknown (e.g. the
    manager is used outside SGLang's distributed environment).
    """
    if _is_gb200_gb300_or_vr200():
        return
    if node_count is None or node_count == 1:
        return
    raise RuntimeError(
        "The 'flashinfer_nvlink_one_sided' all2all backend needs every rank on "
        "one physical node on this GPU, which has no NVLink between hosts, but "
        f"the ranks span {node_count} nodes. Switch to the "
        "'allgather_reducescatter' backend manually with "
        "`--moe-a2a-backend none`."
    )
