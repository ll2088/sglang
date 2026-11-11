"""Host-memory headroom bounded by the process's visible cgroup hierarchy."""

from __future__ import annotations

import logging
import math
import mmap
import os
import re
from pathlib import Path, PurePosixPath
from stat import S_ISDIR
from typing import Optional

import psutil
import torch

logger = logging.getLogger(__name__)

# Max bytes per cudaHostRegister call; matches the HiSparse limit.
HOST_REGISTER_CHUNK_BYTES = 256 << 30

HUGEPAGE_SIZES = {"2MB": 2 << 20, "1GB": 1 << 30}

_MAP_HUGETLB = 0x40000
_MAP_HUGE_SHIFT = 26


def _unescape_mount_path(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def _memory_stat(directory: Path) -> dict[str, int]:
    lines = (directory / "memory.stat").read_text().splitlines()
    return {key: int(value) for key, value in map(str.split, lines)}


def _has_protected_descendants(directory: Path) -> bool:
    """Treat configured or unreadable descendant memory.min as protected."""
    pending = [directory]
    try:
        while pending:
            for child in pending.pop().iterdir():
                if not S_ISDIR(child.stat().st_mode):
                    continue
                if int((child / "memory.min").read_text()) != 0:
                    return True
                pending.append(child)
    except (OSError, ValueError):
        # Unknown protection only disables cache credit, not the cgroup bound.
        return True
    return False


def _cgroup_memory_headroom(proc_root: Path = Path("/proc")) -> int | None:
    memberships = {}
    try:
        cgroups = (proc_root / "self/cgroup").read_text()
        mounts = (proc_root / "self/mountinfo").read_text()
    except FileNotFoundError:
        # Non-Linux systems need not expose procfs.
        return None
    for line in cgroups.splitlines():
        _, controllers, path = line.split(":", 2)
        if not controllers:
            memberships["cgroup2"] = PurePosixPath(path)
        elif "memory" in controllers.split(","):
            memberships["cgroup"] = PurePosixPath(path)

    headroom = None
    resolved = False
    for line in mounts.splitlines():
        before, after = line.split(" - ", 1)
        filesystem, _, options = after.split()[:3]
        if filesystem not in memberships:
            continue
        if filesystem == "cgroup" and "memory" not in options.split(","):
            continue
        fields = before.split()
        root = PurePosixPath(_unescape_mount_path(fields[3]))
        mount = Path(_unescape_mount_path(fields[4]))
        membership = memberships[filesystem]
        if membership.is_relative_to(root):
            relative = membership.relative_to(root)
        elif root != PurePosixPath("/"):
            # A cgroup namespace can expose membership relative to its root,
            # while mountinfo still identifies the host-side subtree.
            relative = membership.relative_to("/")
        else:
            continue
        if ".." in relative.parts:
            raise ValueError(f"Cannot resolve cgroup memory path: {membership}")
        directory = mount / relative
        if not directory.is_dir():
            continue
        resolved = True
        limits = (
            ("memory.max", "memory.high")
            if filesystem == "cgroup2"
            else ("memory.limit_in_bytes",)
        )
        usage_name = (
            "memory.current" if filesystem == "cgroup2" else "memory.usage_in_bytes"
        )
        # File LRU pages, as MemAvailable counts; shmem sits on the anon LRU.
        # v1's unprefixed counters omit descendants, unlike its usage.
        page_cache_keys = (
            ("active_file", "inactive_file")
            if filesystem == "cgroup2"
            else ("total_active_file", "total_inactive_file")
        )
        while True:
            usage = None
            for name in limits:
                try:
                    value = (directory / name).read_text().strip()
                except FileNotFoundError:
                    # The hierarchy root may not have memory controller files.
                    continue
                if value == "max":
                    continue
                limit = int(value)
                # Do not silently ignore an unreadable usage file for a known
                # limit: falling back to host RAM could overrun the container.
                if usage is None:
                    usage = int((directory / usage_name).read_text())
                    stat = _memory_stat(directory)
                    page_cache = sum(stat[key] for key in page_cache_keys)
                    # Protection is relative to the reclaim target: its own min
                    # does not prevent reclaim, but descendants' min can.
                    if (
                        filesystem == "cgroup2"
                        and page_cache > 0
                        and _has_protected_descendants(directory)
                    ):
                        page_cache = 0
                    usage = max(0, usage - page_cache)
                remaining = max(0, limit - usage)
                headroom = remaining if headroom is None else min(headroom, remaining)
            if directory == mount:
                break
            directory = directory.parent
    if memberships and not resolved:
        raise RuntimeError(
            "Cannot locate the process memory cgroup in mounted cgroup filesystems"
        )
    return headroom


def parse_hugepage_size(value: object) -> Optional[int]:
    """Parse a HugeTLB size key; ``None`` or ``"none"`` disables it."""
    key = str(value or "none").upper()
    if key == "NONE":
        return None
    if key not in HUGEPAGE_SIZES:
        raise ValueError(
            f"Unsupported hugepage size {value!r}; "
            f"expected one of {sorted(HUGEPAGE_SIZES)} or none"
        )
    return HUGEPAGE_SIZES[key]


def alloc_host_buffer(
    nbytes: int, hugepage_size: Optional[int] = None
) -> torch.Tensor:
    """Map one host buffer of whole backing pages and return an int8 view.

    HugeTLB pages never fall back to ordinary pages: a shortfall raises.
    """
    page_size = hugepage_size or mmap.PAGESIZE
    nbytes = max(nbytes, 1)
    nbytes = (nbytes + page_size - 1) // page_size * page_size
    kind = f"{page_size >> 20} MiB HugeTLB" if hugepage_size else "ordinary"
    flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
    if hugepage_size:
        flags |= _MAP_HUGETLB | (hugepage_size.bit_length() - 1) << _MAP_HUGE_SHIFT
    try:
        buf = mmap.mmap(-1, nbytes, flags=flags)
    except OSError as e:
        raise RuntimeError(
            f"Cannot map {nbytes / 2**30:.2f} GiB of {kind} pages: {e}. "
            "Reserve the pages on the host and request them for the container."
        ) from e
    # Keep pinned pages out of fork() children.
    try:
        buf.madvise(mmap.MADV_DONTFORK)
    except (AttributeError, OSError) as e:
        logger.warning("MADV_DONTFORK failed: %s", e)
    if not hugepage_size and os.environ.get("THP_MEM_ALLOC_ENABLE") == "1":
        try:
            buf.madvise(mmap.MADV_HUGEPAGE)
        except (AttributeError, OSError) as e:
            logger.warning("MADV_HUGEPAGE failed: %s", e)
    # Populate up front so a pool or cgroup shortfall fails here, not as SIGBUS.
    try:
        buf.write(b"\x00" * nbytes)
    except (OSError, ValueError) as e:
        raise RuntimeError(
            f"Cannot populate {nbytes / 2**30:.2f} GiB of {kind} pages: "
            f"{e}. The container's memory or hugetlb limit is too small."
        ) from e
    return torch.frombuffer(buf, dtype=torch.int8)


def split_host_buffer(
    buffer: torch.Tensor, sizes: list[int]
) -> list[torch.Tensor]:
    """Split one host buffer into contiguous, non-overlapping views."""
    total = sum(sizes)
    if total > buffer.nbytes:
        raise ValueError(
            f"Host tensor sizes {total} exceed the shared host buffer "
            f"{buffer.nbytes}."
        )
    views: list[torch.Tensor] = []
    offset = 0
    for size in sizes:
        views.append(buffer[offset : offset + size])
        offset += size
    return views


def pin_tensor(tensor: torch.Tensor) -> None:
    """Register one CPU range, bypassing the caching allocator's rounding."""
    err = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)
    if err.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {err}")


def unpin_tensor(tensor: torch.Tensor) -> None:
    """Unregister exactly the range passed to pin_tensor."""
    cudart = torch.cuda.cudart()
    err = cudart.cudaHostUnregister(tensor.data_ptr())
    if err.value != 0:
        cudart.cudaGetLastError()
        raise RuntimeError(f"cudaHostUnregister failed (code={err.value}).")


def host_registration_ranges(
    tensor_sizes: list[int],
    num_blocks: int,
    total_size: int,
    max_chunk_bytes: int = HOST_REGISTER_CHUNK_BYTES,
    *,
    page_size: int = mmap.PAGESIZE,
) -> tuple[tuple[int, int], ...]:
    """Split a shared pool without bisecting any tensor's host block.

    Keep registration boundaries aligned to the backing pages. CUDA batch
    copies reject descriptors spanning adjacent registrations.
    """
    if total_size % page_size:
        raise ValueError("Host pool must be page aligned.")
    if max_chunk_bytes < page_size:
        raise ValueError("Host registration chunks must span at least one page.")

    tensors: list[tuple[int, int, int]] = []
    tensor_start = 0
    for tensor_size in tensor_sizes:
        if tensor_size % num_blocks:
            raise ValueError(
                f"Host tensor size {tensor_size} is not divisible by "
                f"{num_blocks} blocks."
            )
        tensor_end = tensor_start + tensor_size
        tensors.append((tensor_start, tensor_end, tensor_size // num_blocks))
        tensor_start = tensor_end
    if tensor_start > total_size:
        raise ValueError("Host tensors exceed the shared host pool.")

    def last_safe_boundary(lower: int, upper: int) -> Optional[int]:
        best = None
        for start, end, block_bytes in tensors:
            first_block = max(0, (lower - start) // block_bytes + 1)
            last_block = min(num_blocks, (min(upper, end) - start) // block_bytes)
            if first_block > last_block:
                continue
            divisor = math.gcd(block_bytes, page_size)
            if start % divisor:
                continue
            period = page_size // divisor
            if period == 1:
                residue = 0
            else:
                # Find block boundaries that are also backing-page boundaries.
                residue = (
                    (-start // divisor) * pow(block_bytes // divisor, -1, period)
                ) % period
            block = last_block - (last_block - residue) % period
            if block >= first_block:
                candidate = start + block * block_bytes
                best = candidate if best is None else max(best, candidate)

        if upper >= tensor_start:
            candidate = upper // page_size * page_size
            if candidate >= tensor_start and candidate > lower:
                best = candidate if best is None else max(best, candidate)
        return best

    ranges: list[tuple[int, int]] = []
    start = 0
    while total_size - start > max_chunk_bytes:
        end = last_safe_boundary(start, start + max_chunk_bytes)
        if end is None:
            raise ValueError(
                "Cannot split host registration without bisecting a block."
            )
        ranges.append((start, end))
        start = end
    ranges.append((start, total_size))
    return tuple(ranges)


def pin_tensor_chunked(
    tensor: torch.Tensor,
    tensor_sizes: list[int],
    num_blocks: int,
    *,
    page_size: int = mmap.PAGESIZE,
) -> tuple[tuple[int, int], ...]:
    """Register a shared host pool in page-aligned, block-safe chunks."""
    ranges = host_registration_ranges(
        tensor_sizes, num_blocks, tensor.nbytes, page_size=page_size
    )
    base = tensor.data_ptr()
    cudart = torch.cuda.cudart()
    registered: list[tuple[int, int]] = []
    try:
        for start, end in ranges:
            err = cudart.cudaHostRegister(base + start, end - start, 0)
            if err.value != 0:
                raise RuntimeError(f"cudaHostRegister failed: {err}")
            registered.append((start, end))
    except Exception:
        for start, end in registered:
            cudart.cudaHostUnregister(base + start)
        raise
    return tuple(registered)


def unpin_tensor_chunked(
    tensor: torch.Tensor, ranges: tuple[tuple[int, int], ...]
) -> None:
    """Unregister exactly the ranges returned by pin_tensor_chunked."""
    base = tensor.data_ptr()
    cudart = torch.cuda.cudart()
    for start, _ in ranges:
        err = cudart.cudaHostUnregister(base + start)
        if err.value != 0:
            cudart.cudaGetLastError()
            raise RuntimeError(f"cudaHostUnregister failed (code={err.value}).")


def available_host_memory_bytes(*, allow_cgroup_fallback: bool = False) -> int:
    """Allocatable RAM, counting reclaimable page cache as free."""
    available = psutil.virtual_memory().available
    try:
        cgroup_headroom = _cgroup_memory_headroom()
    except (OSError, RuntimeError) as exc:
        if not allow_cgroup_fallback:
            raise RuntimeError(
                "Cannot automatically size HiCache host memory; "
                "set --hicache-size explicitly (GB per rank), leaving room "
                "for other memory use within the container limit."
            ) from exc
        logger.warning(
            "Cannot determine cgroup memory headroom: %s. Checking explicitly "
            "sized HiCache pools against host available memory; ensure all "
            "local ranks' pools fit within the container memory limit.",
            exc,
        )
        return available
    if cgroup_headroom is not None:
        logger.info(
            "HiCache memory headroom: host %.1f GiB, cgroup %.1f GiB",
            available / 1024**3,
            cgroup_headroom / 1024**3,
        )
        available = min(available, cgroup_headroom)
    return available
