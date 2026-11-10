"""Capability-probe / fallback tests for the opt-in FlyDSL FP8 MLA prefill.

These tests are deliberately lightweight: they do not import the full
``sglang`` package. They exercise the pure-Python capability predicate that
gates the opt-in backend, so a missing FlyDSL/AITER install or a non-gfx950
device must fall back to the default prefill path without raising.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# These tests load the capability predicate by file path so they do not need a
# full sglang install. torch/triton are optional here: skip cleanly when absent
# instead of erroring during collection.
torch = pytest.importorskip("torch", reason="torch required for FP8 quant tests")


def _load_module(rel_path, name):
    here = Path(__file__).resolve()
    repo_root = here.parents[5]
    target = repo_root / rel_path
    if not target.is_file():
        return None
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, target)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001
        sys.modules.pop(name, None)
        return None
    return module


def test_module_loader_handles_missing_path():
    """The by-path loader used by these tests must not raise on a bad path."""
    assert _load_module(Path("does/not/exist.py"), "_sglang_missing_mod") is None


_caps = _load_module(
    Path("python/sglang/kernels/ops/attention/mla_flydsl_fp8_caps.py"),
    "_sglang_mla_flydsl_fp8_caps_under_test",
)

pytestmark = pytest.mark.skipif(
    _caps is None, reason="mla_flydsl_fp8_caps unavailable"
)


def test_default_off_never_selects_flydsl():
    """With the opt-in switch off the backend is never selected."""
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=False, gfx="gfx950", flydsl_available=True
    )


def test_requires_gfx950():
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx="gfx942", flydsl_available=True
    )
    assert _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx="gfx950", flydsl_available=True
    )


def test_requires_flydsl():
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx="gfx950", flydsl_available=False
    )


def test_unknown_gfx_falls_back():
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx=None, flydsl_available=True
    )


def test_fallback_reason_is_explicit():
    reason = _caps.flydsl_fp8_fallback_reason(
        enabled=True, gfx="gfx942", flydsl_available=True
    )
    assert reason is not None and "gfx950" in reason
    assert (
        _caps.flydsl_fp8_fallback_reason(
            enabled=True, gfx="gfx950", flydsl_available=True
        )
        is None
    )
