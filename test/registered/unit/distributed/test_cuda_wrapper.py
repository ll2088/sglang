import io

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

from sglang.srt.distributed.device_communicators import cuda_wrapper
from sglang.srt.distributed.device_communicators import pynccl as srt_pynccl


def test_pynccl_require_usable_communicator_rejects_disabled():
    """A disabled PyNccl communicator must fail fast instead of no-op."""
    comm = srt_pynccl.PyNcclCommunicator.__new__(srt_pynccl.PyNcclCommunicator)
    comm.disabled = True
    comm.available = False
    comm.world_size = 2
    with pytest.raises(RuntimeError, match="disabled"):
        srt_pynccl._require_usable_communicator(comm)


def test_pynccl_require_usable_communicator_accepts_enabled():
    comm = srt_pynccl.PyNcclCommunicator.__new__(srt_pynccl.PyNcclCommunicator)
    comm.disabled = False
    comm.available = True
    comm.world_size = 2
    assert srt_pynccl._require_usable_communicator(comm) is comm


def test_pynccl_wrapper_missing_library_raises(monkeypatch):
    """NCCLLibrary must raise (not silently disable) when the .so is missing."""
    from sglang.srt.distributed.device_communicators import pynccl_wrapper

    def _boom(*args, **kwargs):
        raise OSError("libnccl not found")

    monkeypatch.setattr(pynccl_wrapper.ctypes, "CDLL", _boom)
    with pytest.raises(OSError, match="libnccl"):
        pynccl_wrapper.NCCLLibrary(so_file="/nonexistent/libnccl.so.2")


def test_xpu_communicator_disabled_off_xpu(monkeypatch):
    """XpuCommunicator must mark itself disabled when not on XPU."""
    from sglang.srt.distributed.device_communicators import xpu_communicator

    monkeypatch.setattr(xpu_communicator, "is_xpu", lambda: False)
    comm = xpu_communicator.XpuCommunicator(group=None)
    assert comm.disabled is True


def test_multimodal_pynccl_require_usable_communicator_rejects_disabled():
    """The multimodal_gen PyNccl wrapper must also fail fast when disabled."""
    from sglang.multimodal_gen.runtime.distributed.device_communicators import (
        pynccl as mm_pynccl,
    )

    comm = mm_pynccl.PyNcclCommunicator.__new__(mm_pynccl.PyNcclCommunicator)
    comm.disabled = True
    comm.available = False
    comm.world_size = 2
    with pytest.raises(RuntimeError, match="disabled"):
        mm_pynccl._require_usable_communicator(comm)


def test_find_loaded_library_prefers_real_cudart_over_tilelang_stub(monkeypatch):
    maps = """\
7f000000-7f010000 r-xp 00000000 00:00 0 /site-packages/tilelang/lib/libcudart_stub.so
7f020000-7f030000 r-xp 00000000 00:00 0 /cuda/lib64/libcudart.so.13
"""

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(maps))

    assert (
        cuda_wrapper.find_loaded_library("libcudart") == "/cuda/lib64/libcudart.so.13"
    )


def test_find_loaded_library_strips_deleted_suffix(monkeypatch):
    maps = """\
7f020000-7f030000 r-xp 00000000 00:00 0 /cuda/lib64/libcudart.so.13 (deleted)
"""

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(maps))

    assert (
        cuda_wrapper.find_loaded_library("libcudart") == "/cuda/lib64/libcudart.so.13"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
