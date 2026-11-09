import importlib.metadata
import subprocess
import sys

import pytest


def _parse_version(version):
    parts = []
    for piece in version.split("."):
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
    version = importlib.metadata.version("cuda-python")
    assert _parse_version(version) >= (13, 5, 0), version
    import sgl_kernel  # noqa: F401


def test_load_utils_version_probe():
    from sgl_kernel import load_utils

    cuda_home = load_utils._find_cuda_home()
    assert isinstance(cuda_home, str)
    assert cuda_home

    capability = load_utils._get_compute_capability()
    assert capability is None or isinstance(capability, int)


def test_check_env_reports_cuda_version(monkeypatch):
    import sglang.check_env as check_env

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
