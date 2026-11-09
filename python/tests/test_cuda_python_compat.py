import importlib
import importlib.metadata
import subprocess
import sys

import pytest

# Ensure the test module can be collected even when sglang/sgl_kernel are not
# importable in the current environment; individual tests skip instead.
pytestmark = pytest.mark.importorskip("cuda.bindings")


def _cuda_python_version():
    try:
        return importlib.metadata.version("cuda-python")
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_cuda_python_13_5():
    version = _cuda_python_version()
    if version is None:
        pytest.skip("cuda-python is not installed in this environment")
    if _parse_version(version) < (13, 5, 0):
        pytest.skip(f"cuda-python {version} is older than the required 13.5.0")
    return version


def _import_or_skip(module_name):
    """Import a module, skipping the test if the environment cannot provide it."""
    try:
        return importlib.import_module(module_name)
    except BaseException as exc:  # pragma: no cover - environment dependent
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        pytest.skip(f"{module_name} is not importable in this environment: {exc}")


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


def test_cuda_python_version_and_bindings_import():
    version = _require_cuda_python_13_5()
    assert _parse_version(version) >= (13, 5, 0), version
    # The bindings package must expose the modules this repo relies on.
    bindings = _import_or_skip("cuda.bindings")
    assert bindings is not None


def test_sgl_kernel_import():
    _require_cuda_python_13_5()
    # sgl_kernel is a compiled CUDA extension; it is absent in CPU-only CI
    # images, so skip rather than fail when it cannot be imported. Importing
    # it can raise non-ImportError exceptions (e.g. missing libcudart), so
    # guard the import explicitly instead of relying on importorskip.
    sgl_kernel = _import_or_skip("sgl_kernel")
    assert sgl_kernel is not None


def test_load_utils_version_probe():
    _require_cuda_python_13_5()
    load_utils = _import_or_skip("sgl_kernel.load_utils")

    cuda_home = load_utils._find_cuda_home()
    assert isinstance(cuda_home, str)
    assert cuda_home

    capability = load_utils._get_compute_capability()
    assert capability is None or isinstance(capability, int)

    # _preload_cuda_library must be a no-op-safe probe even without a GPU.
    load_utils._preload_cuda_library()


def test_check_env_reports_cuda_version(monkeypatch):
    version = _require_cuda_python_13_5()
    check_env = _import_or_skip("sglang.check_env")

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
    # cuda-python is part of check_env.PACKAGE_LIST; verify the probe reports
    # the installed version through importlib.metadata.
    assert versions.get("cuda-python") == version

    result = subprocess.run(
        [sys.executable, "-m", "sglang.check_env"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and (
        "No module named" in result.stderr or "ModuleNotFoundError" in result.stderr
    ):
        pytest.skip(f"sglang is not importable in a subprocess: {result.stderr}")
    assert result.returncode == 0, result.stderr
    assert "Python" in result.stdout
    assert "cuda-python" in result.stdout
