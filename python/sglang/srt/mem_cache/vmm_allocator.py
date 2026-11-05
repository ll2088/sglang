# Copyright 2025 SGLang Team
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
"""Minimal VMM-backed growable device buffer.

This module is a small, self-contained port of the two-phase sizing idea from
vLLM's extensible KV cache: reserve a virtual address range up front, then map
physical pages into it once the final size is known. Addresses never move, so
tensors and captured CUDA graphs that view the reservation stay valid.

The implementation is intentionally conservative: any failure (missing driver
symbols, unsupported platform, mapping error) is reported through
``vmm_unavailable_reason`` / raised as ``VmmUnavailableError`` so callers can
fall back to the existing allocation path.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional

import torch

__all__ = [
    "VmmUnavailableError",
    "VmmBuffer",
    "vmm_unavailable_reason",
    "is_vmm_available",
]


class VmmUnavailableError(RuntimeError):
    """Raised when the driver VMM API cannot be used on this host."""


# CUDA driver constants (shared with HIP for the subset we use).
_CUDA_SUCCESS = 0
_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3

# Commit in units of at least 2 MiB to keep granule bookkeeping small even on
# ROCm, where the driver reports a 4 KiB granularity.
_MIN_GRANULARITY = 2 << 20


class _CuMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CuMemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _CuMemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CuMemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _CuMemAllocFlags),
    ]


class _CuMemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CuMemLocation), ("flags", ctypes.c_int)]


class _CudaArrayInterface:
    """Minimal CUDA array interface over a raw device pointer.

    ``torch.as_tensor`` consumes ``__cuda_array_interface__`` and returns a
    view that does not own the memory, which is what the VMM reservation
    needs. The interface is read-only from torch's perspective; callers must
    not free the reservation while a view is alive.
    """

    def __init__(self, ptr: int, nbytes: int, dtype: torch.dtype) -> None:
        self._ptr = ptr
        self._nbytes = nbytes
        self._dtype = dtype

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self._nbytes // self._dtype.itemsize,),
            "typestr": _torch_dtype_to_typestr(self._dtype),
            "data": (self._ptr, False),
            "version": 3,
            "stream": None,
        }


def _torch_dtype_to_typestr(dtype: torch.dtype) -> str:
    """Map a torch dtype to the NumPy-style typestr used by the interface."""
    mapping = {
        torch.float16: "<f2",
        torch.bfloat16: "<V2",
        torch.float32: "<f4",
        torch.float64: "<f8",
        torch.int8: "|i1",
        torch.uint8: "|u1",
        torch.int16: "<i2",
        torch.int32: "<i4",
        torch.int64: "<i8",
        torch.bool: "|b1",
    }
    try:
        return mapping[dtype]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unsupported dtype for VMM view: {dtype}") from exc


_DRIVER: Optional[ctypes.CDLL] = None
_DRIVER_ERROR: Optional[str] = None


def _find_loaded_library(lib_name: str) -> Optional[str]:
    """Return the path of an already-mapped shared object named lib_name.

    Preferring the copy already in the process avoids loading a second driver
    runtime (e.g. the one bundled with a ROCm torch wheel).
    """
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "/" not in line:
                    continue
                path = line[line.index("/") :].strip()
                if os.path.basename(path).startswith(f"{lib_name}.so"):
                    return path
    except (OSError, ValueError):
        return None
    return None


def _load_driver() -> ctypes.CDLL:
    global _DRIVER, _DRIVER_ERROR
    if _DRIVER is not None:
        return _DRIVER
    if _DRIVER_ERROR is not None:
        raise VmmUnavailableError(_DRIVER_ERROR)
    if not torch.cuda.is_available():
        _DRIVER_ERROR = "CUDA is not available"
        raise VmmUnavailableError(_DRIVER_ERROR)
    if path := _find_loaded_library("libcuda"):
        try:
            _DRIVER = ctypes.CDLL(path)
            return _DRIVER
        except OSError:
            pass
    candidates = ("libcuda.so.1", "libcuda.so")
    last_err: Optional[str] = None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
        except OSError as exc:  # pragma: no cover - platform dependent
            last_err = str(exc)
            continue
        _DRIVER = lib
        return lib
    _DRIVER_ERROR = f"could not load CUDA driver: {last_err}"
    raise VmmUnavailableError(_DRIVER_ERROR)


def _check(result: int, what: str) -> None:
    if result != _CUDA_SUCCESS:
        raise VmmUnavailableError(f"{what} failed with CUresult={result}")


def vmm_unavailable_reason() -> Optional[str]:
    """Return a human-readable reason when VMM cannot be used, else None."""
    if os.environ.get("SGLANG_DISABLE_VMM", "0") == "1":
        return "SGLANG_DISABLE_VMM=1"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    # The driver API is only meaningful once a CUDA context exists; probing
    # before initialization would report a false negative on some hosts.
    try:
        torch.cuda.current_device()
    except Exception as exc:  # pragma: no cover - driver dependent
        return f"CUDA context unavailable: {exc}"
    try:
        lib = _load_driver()
    except VmmUnavailableError as exc:
        return str(exc)
    for symbol in (
        "cuMemAddressReserve",
        "cuMemAddressFree",
        "cuMemCreate",
        "cuMemMap",
        "cuMemUnmap",
        "cuMemRelease",
        "cuMemSetAccess",
        "cuMemGetAllocationGranularity",
    ):
        if not hasattr(lib, symbol):
            return f"driver is missing {symbol}"
    return None


def is_vmm_available() -> bool:
    return vmm_unavailable_reason() is None


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class VmmBuffer:
    """A device virtual address reservation with lazily mapped physical pages.

    The buffer exposes a stable ``data_ptr`` for the whole reservation. Callers
    build tensor views over it once and then call :meth:`commit` to map more
    physical memory as the final size becomes known.
    """

    def __init__(self, max_bytes: int, device_index: int) -> None:
        reason = vmm_unavailable_reason()
        if reason is not None:
            raise VmmUnavailableError(reason)
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._lib = _load_driver()
        self.device_index = device_index
        self._freed = False

        # Ensure the CUDA context exists before touching the driver API.
        torch.cuda.current_device()

        self.granularity = self._query_granularity(device_index)
        self.reserved_size = _round_up(max_bytes, self.granularity)
        self.base_ptr = self._reserve(self.reserved_size)
        self._mapped_granules: set[int] = set()
        self._handles: list[tuple[int, int, int]] = []

    # -- driver helpers ---------------------------------------------------

    def _query_granularity(self, device_index: int) -> int:
        prop = _CuMemAllocationProp()
        prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device_index
        granule = ctypes.c_size_t(0)
        _check(
            self._lib.cuMemGetAllocationGranularity(
                ctypes.byref(granule),
                ctypes.byref(prop),
                _CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            ),
            "cuMemGetAllocationGranularity",
        )
        return _round_up(max(granule.value, _MIN_GRANULARITY), granule.value)

    def _reserve(self, size: int) -> int:
        ptr = ctypes.c_ulonglong(0)
        _check(
            self._lib.cuMemAddressReserve(
                ctypes.byref(ptr), size, 0, 0, 0
            ),
            "cuMemAddressReserve",
        )
        return ptr.value

    def _make_prop(self) -> _CuMemAllocationProp:
        prop = _CuMemAllocationProp()
        prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device_index
        return prop

    def _map_granule(self, granule_index: int) -> None:
        prop = self._make_prop()
        handle = ctypes.c_ulonglong(0)
        _check(
            self._lib.cuMemCreate(
                ctypes.byref(handle), self.granularity, ctypes.byref(prop), 0
            ),
            "cuMemCreate",
        )
        va = self.base_ptr + granule_index * self.granularity
        try:
            _check(
                self._lib.cuMemMap(va, self.granularity, handle.value, 0),
                "cuMemMap",
            )
            access = _CuMemAccessDesc()
            access.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = self.device_index
            access.flags = _CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _check(
                self._lib.cuMemSetAccess(
                    va, self.granularity, ctypes.byref(access), 1
                ),
                "cuMemSetAccess",
            )
        except Exception:
            self._lib.cuMemRelease(handle.value)
            raise
        self._handles.append((handle.value, va, self.granularity))
        self._mapped_granules.add(granule_index)

    # -- public API -------------------------------------------------------

    @property
    def committed_bytes(self) -> int:
        return len(self._mapped_granules) * self.granularity

    @property
    def num_committed_granules(self) -> int:
        return len(self._mapped_granules)

    def commit(self, num_bytes: int) -> int:
        """Map physical pages so that at least ``num_bytes`` are backed.

        Returns the number of bytes actually committed (a multiple of the
        granularity). Committing fewer bytes than already committed is a no-op,
        which makes repeated calls idempotent.
        """
        if self._freed:
            raise RuntimeError("VmmBuffer has been freed")
        if num_bytes < 0:
            raise ValueError("num_bytes must be non-negative")
        if num_bytes > self.reserved_size:
            raise ValueError(
                f"requested {num_bytes} bytes exceeds reserved {self.reserved_size}"
            )
        target = _round_up(num_bytes, self.granularity) // self.granularity
        # Committing fewer bytes than already committed is a no-op, which
        # keeps repeated calls idempotent and monotonic.
        if target <= len(self._mapped_granules):
            return self.committed_bytes
        for granule_index in range(target):
            if granule_index not in self._mapped_granules:
                self._map_granule(granule_index)
        return self.committed_bytes

    def as_tensor(self, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        """Return a tensor viewing the reservation without taking ownership.

        The view is built over the stable ``base_ptr`` of the reservation, so
        it stays valid as more granules are committed. The returned tensor does
        not own the memory; :meth:`free` must be called explicitly.
        """
        if self._freed:
            raise RuntimeError("VmmBuffer has been freed")
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * torch.tensor([], dtype=dtype).element_size()
        if nbytes > self.reserved_size:
            raise ValueError(
                f"tensor needs {nbytes} bytes but only {self.reserved_size} reserved"
            )
        # Build a tensor over the reservation without copying or taking
        # ownership. ``torch.frombuffer`` only accepts host buffers, so wrap
        # the device pointer with the CUDA array interface, which every
        # supported torch version understands and which keeps the
        # reservation's stable base pointer.
        device = torch.device("cuda", self.device_index)
        # ``torch.cuda.caching_allocator_alloc`` would take ownership, so we
        # deliberately avoid it. ``torch.as_tensor`` on a CUDA array interface
        # object creates a view over the existing pointer. The interface is
        # passed positionally and the device is inferred from it, which is the
        # form supported across torch versions.
        flat = torch.as_tensor(
            _CudaArrayInterface(self.base_ptr, nbytes, dtype),
        )
        if flat.device != device:
            raise VmmUnavailableError(
                f"VMM view landed on {flat.device}, expected {device}"
            )
        return flat.view(tuple(shape))

    def free(self) -> None:
        if self._freed:
            return
        self._freed = True
        for handle, va, size in self._handles:
            try:
                self._lib.cuMemUnmap(va, size)
            finally:
                self._lib.cuMemRelease(handle)
        self._handles.clear()
        self._mapped_granules.clear()
        if self.base_ptr:
            self._lib.cuMemAddressFree(self.base_ptr, self.reserved_size)
            self.base_ptr = 0

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.free()
        except Exception:
            pass
