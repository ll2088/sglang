import importlib
import importlib.metadata

import pytest


CUDA_PYTHON_VERSION = "13.5.0"

SGL_KERNEL_SUBMODULES = [
    "sgl_kernel",
    "sgl_kernel.memory",
    "sgl_kernel.load_utils",
    "sgl_kernel.utils",
    "sgl_kernel.version",
]


def _installed_cuda_python_version():
    try:
        return importlib.metadata.version("cuda-python")
    except importlib.metadata.PackageNotFoundError:
        return None


@pytest.mark.parametrize("module_name", SGL_KERNEL_SUBMODULES)
@pytest.mark.parametrize("binding_name", ["cuda", "cuda.bindings"])
def test_cuda_python_bindings_importable(module_name, binding_name):
    """cuda-python 13.5.0 must expose the bindings used by sgl_kernel."""
    importlib.import_module(binding_name)
    importlib.import_module(module_name)


def test_sgl_kernel_version_matches_cuda_python():
    """sgl_kernel.version must report the installed cuda-python version."""
    installed = _installed_cuda_python_version()
    if installed is None:
        pytest.skip("cuda-python is not installed in this environment")
    assert installed == CUDA_PYTHON_VERSION

    version_module = importlib.import_module("sgl_kernel.version")
    assert version_module.__version__ == installed


def test_check_env_reports_cuda_python_version():
    """sglang.check_env must surface the cuda-python version."""
    installed = _installed_cuda_python_version()
    if installed is None:
        pytest.skip("cuda-python is not installed in this environment")

    check_env = importlib.import_module("sglang.check_env")
    env = check_env._get_env()
    versions = env.get_package_versions()
    assert versions.get("cuda-python") == CUDA_PYTHON_VERSION
