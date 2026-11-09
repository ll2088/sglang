import importlib.metadata
import subprocess
import sys

import pytest


def _cuda_python_version():
    try:
        return importlib.metadata.version("cuda-python")
    except importlib.metadata.PackageNotFoundError:
        return None


_CUDA_PYTHON_VERSION = _cuda_python_version()

pytestmark = pytest.mark.skipif(
    _CUDA_PYTHON_VERSION is None,
    reason="cuda-python is not installed in this environment",
)


def _require_cuda_python_13_5():
    if _CUDA_PYTHON_VERSION is None:
        pytest.skip("cuda-python is not installed in this environment")
    if _parse_version(_CUDA_PYTHON_VERSION) < (13, 5, 0):
        pytest.skip(
            f"cuda-python {_CUDA_PYTHON_VERSION} is older than the required 13.5.0"
        )


def _parse_version(version):
    # Strip pre-release/local suffixes (e.g. "13.5.0rc1", "13.5.0+cu13")
    # so the numeric comparison below stays well-defined.
    core = version.split("+", 1)[0]
    for sep in ("rc", "a", "b", "dev", "post"):
        idx = core.find(sep)
        if idx != -1:
            core = core[:idx]
            break
    parts = []
    for piece in core.split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def test_cuda_python_version_and_sgl_kernel_import():
    _require_cuda_python_13_5()
    version = _CUDA_PYTHON_VERSION
    assert _parse_version(version) >= (13, 5, 0), version
    pytest.importorskip("sgl_kernel")


def test_load_utils_version_probe():
    _require_cuda_python_13_5()
    load_utils = pytest.importorskip("sgl_kernel.load_utils")

    cuda_home = load_utils._find_cuda_home()
    assert isinstance(cuda_home, str)
    assert cuda_home

    capability = load_utils._get_compute_capability()
    assert capability is None or isinstance(capability, int)


def test_check_env_reports_cuda_version(monkeypatch):
    _require_cuda_python_13_5()
    check_env = pytest.importorskip("sglang.check_env")

    monkeypatch.setattr(check_env, "is_cuda_v2", lambda: False)
    monkeypatch.setattr(check_env, "is_hip", lambda: False)
    monkeypatch.setattr(check_env, "is_npu", lambda: False)
    monkeypatch.setattr(check_env, "is_musa", lambda: False)
    monkeypatch.setattr(check_env, "is_xpu", lambda: False)
    monkeypatch.setattr(check_env, "is_mps", lambda: False)

    env = check_env._get_env()
    assert isinstance(env, check_env.CPUEnv)
    info = env.get_info()
    assert "CPU Architecture" in info

    versions = env.get_package_versions()
    assert isinstance(versions, dict)

    result = subprocess.run(
        [sys.executable, "-m", "sglang.check_env"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Python" in result.stdout
