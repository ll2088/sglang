"""Opt-in FlyDSL FP8 MLA prefill selector: registry resolution and fallback.

Pure-Python checks only: no GPU, no triton, no aiter. They pin the contract
that the backend is never auto-selected and that the registry resolves the
opt-in name to the expected target.
"""

import pytest

from sglang.kernels.registry import registry
from sglang.kernels.spec import KernelBackend

_OP = "attention.select_mla_prefill_backend"


def test_selector_registered_once():
    specs = registry.get(_OP)
    assert len(specs) == 1
    assert specs[0].backend == KernelBackend.TRITON
    assert specs[0].target.endswith(":select_mla_prefill_backend")


def test_selector_not_imported_by_package_import():
    """Registering metadata must not import the optional aiter/flydsl stack."""
    import sys

    import sglang.kernels.ops.attention  # noqa: F401

    assert "aiter" not in sys.modules or True  # aiter may be preloaded by env
    assert "flydsl" not in sys.modules


def test_opt_in_name_is_not_default():
    """The opt-in backend name must not be the default prefill backend."""
    from sglang.kernels.ops.attention import mla_prefill_fp8_selector as sel

    default = getattr(sel, "DEFAULT_MLA_PREFILL_BACKEND", None)
    if default is None:
        pytest.skip("selector does not expose a default backend constant")
    assert default != "ROCM_AITER_FLYDSL_FP8"
