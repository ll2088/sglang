# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

"""Platform probing and auto-resolution for the MoE all2all backend.

This mirrors the upstream vLLM strategy of resolving ``auto`` against the
hardware topology: ``flashinfer_nvlink_one_sided`` is only picked where it was
measured to beat the ``allgather_reducescatter`` default, i.e. on
GB200/GB300/VR200 (ARM host, compute capability 10.x) and on single-node HGX
B200/B300 (x86 host, compute capability 10.0/10.3).

The probes are pure, cached functions so that unit tests can clear the cache
and simulate different hardware.
"""

from __future__ import annotations

import logging
from functools import cache
from typing import Optional

logger = logging.getLogger(__name__)

# Backends that are resolved by the ``auto`` policy.
AUTO_BACKEND = "auto"
ONE_SIDED_BACKEND = "flashinfer_nvlink_one_sided"
DEFAULT_BACKEND = "none"

# Compute capability families that carry NVLink between hosts over MNNVL.
_ARM_NVL_CAPABILITY_FAMILY = 100
# x86 HGX parts that have the same GPUs but no NVLink between hosts.
_HGX_CAPABILITIES = (100, 103)


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


def _device_capability() -> Optional[tuple[int, int]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return tuple(torch.cuda.get_device_capability(0))
    except Exception:
        return None


def _is_arm_host() -> bool:
    arch = _cpu_arch()
    return arch in ("aarch64", "arm64")


def _is_x86_host() -> bool:
    arch = _cpu_arch()
    return arch in ("x86_64", "amd64")


@cache
def _is_gb200_gb300_or_vr200() -> bool:
    """Whether this is an ARM NVL rack: GB200/GB300 (10.0, 10.3) or VR200
    (10.7). All of them are Blackwell/Rubin 10.x parts on a Grace or Vera host."""
    if not _is_cuda():
        return False
    if not _is_arm_host():
        return False
    capability = _device_capability()
    if capability is None:
        return False
    return capability[0] == _ARM_NVL_CAPABILITY_FAMILY


@cache
def _is_hgx_b200_or_b300() -> bool:
    """Whether this is an x86 HGX B200 (10.0) or HGX B300 (10.3)."""
    if not _is_cuda():
        return False
    if not _is_x86_host():
        return False
    capability = _device_capability()
    if capability is None:
        return False
    return capability in _HGX_CAPABILITIES


def clear_platform_probes() -> None:
    """Clear the cached platform probes (used by tests)."""
    _is_gb200_gb300_or_vr200.cache_clear()
    _is_hgx_b200_or_b300.cache_clear()


def _has_one_sided_support() -> bool:
    try:
        from sglang.srt.utils import has_flashinfer_nvlink_one_sided

        return bool(has_flashinfer_nvlink_one_sided())
    except Exception:
        return False


def prefers_one_sided_all2all(
    *,
    enable_expert_parallel: bool,
    enable_elastic_ep: bool,
    nnodes: int,
    data_parallel_size: int,
    data_parallel_size_local: int,
    distributed_executor_backend: Optional[str],
) -> bool:
    """Whether ``auto`` picks ``flashinfer_nvlink_one_sided``.

    Only where it was measured to beat ``allgather_reducescatter``.
    """
    # The one-sided manager is not wired for elastic EP's stateless groups.
    if not enable_expert_parallel or enable_elastic_ep:
        return False
    if not _has_one_sided_support():
        return False
    # GB200/GB300/VR200 carry NVLink between hosts over MNNVL, so the EP group
    # may span nodes and any data-parallel size is served.
    if _is_gb200_gb300_or_vr200():
        return True
    # HGX has no NVLink between hosts, so the whole deployment must sit on one
    # physical node, placed by SGLang's own launcher or by Ray.
    return (
        nnodes == 1
        and data_parallel_size_local in (0, data_parallel_size)
        and distributed_executor_backend in (None, "mp", "uni", "ray")
        and _is_hgx_b200_or_b300()
    )


def resolve_all2all_backend(
    requested: str,
    *,
    enable_expert_parallel: bool,
    enable_elastic_ep: bool,
    nnodes: int,
    data_parallel_size: int,
    data_parallel_size_local: int,
    distributed_executor_backend: Optional[str],
) -> tuple[str, bool]:
    """Resolve the configured all2all backend.

    Returns:
        A ``(backend, auto_one_sided)`` tuple. ``backend`` is never ``auto``.
    """
    if requested != AUTO_BACKEND:
        return requested, False
    if prefers_one_sided_all2all(
        enable_expert_parallel=enable_expert_parallel,
        enable_elastic_ep=enable_elastic_ep,
        nnodes=nnodes,
        data_parallel_size=data_parallel_size,
        data_parallel_size_local=data_parallel_size_local,
        distributed_executor_backend=distributed_executor_backend,
    ):
        logger.info(
            "all2all backend 'auto' resolved to '%s' for this platform.",
            ONE_SIDED_BACKEND,
        )
        return ONE_SIDED_BACKEND, True
    return DEFAULT_BACKEND, False


def require_single_node_without_mnnvl(node_count: Optional[int], auto: bool) -> None:
    """Fail clearly if one-sided all2all would span hosts without MNNVL.

    GB200/GB300/VR200 connect hosts over MNNVL. On other hardware (HGX B200 and
    B300) the NVLink domain ends at the node, so every rank must share one host.
    """
    if _is_gb200_gb300_or_vr200():
        return
    if node_count is None or node_count == 1:
        return
    origin = (
        " It was picked by default (moe_a2a_backend='auto'), which assumes a "
        "single physical node."
        if auto
        else ""
    )
    raise RuntimeError(
        "The 'flashinfer_nvlink_one_sided' all2all backend needs every rank on "
        "one physical node on this GPU, which has no NVLink between hosts, but "
        f"the ranks span {node_count} nodes.{origin} Switch to the "
        "'allgather_reducescatter' backend manually with "
        "`--moe-a2a-backend none`."
    )
