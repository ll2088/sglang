"""CPU-only tests for the opt-in FlyDSL FP8 MLA prefill selector.

These run without a GPU, torch or aiter so the import / registration / opt-in
contracts can be checked in any CI environment.
"""

import sys

import pytest

from sglang.kernels.ops.attention import mla_prefill_fp8_selector as sel
from sglang.kernels.registry import registry
from sglang.kernels.spec import KernelBackend


def test_import_smoke():
    import sglang.kernels.ops.attention as m

    assert m is not None


def test_kernels_registered():
    for fn in ("select_mla_prefill_backend", "resolve_mla_prefill_kernel"):
        spec = registry.get_backend(
            f"attention.{fn}", KernelBackend.TRITON
        )
        assert spec.target.endswith(f":{fn}")


def test_default_off(monkeypatch):
    monkeypatch.delenv(sel.OPT_IN_ENV, raising=False)
    assert sel.select_mla_prefill_backend(numel=1024) == ""
    assert sel.resolve_mla_prefill_kernel(numel=1024) is None


def test_explicit_off(monkeypatch):
    monkeypatch.setenv(sel.OPT_IN_ENV, "0")
    assert sel.select_mla_prefill_backend(numel=1024) == ""


def test_opt_in_without_aiter_falls_back(monkeypatch):
    monkeypatch.setenv(sel.OPT_IN_ENV, sel.BACKEND_NAME)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", None)
    assert sel.select_mla_prefill_backend(numel=1024) == ""


def test_opt_in_rejects_empty_and_oversized(monkeypatch):
    monkeypatch.setenv(sel.OPT_IN_ENV, sel.BACKEND_NAME)
    monkeypatch.setattr(sel, "_on_gfx950", lambda: True)
    monkeypatch.setattr(sel, "_aiter_flydsl_fp8_available", lambda: True)
    assert sel.select_mla_prefill_backend(numel=0) == ""
    assert sel.select_mla_prefill_backend(numel=sel.FLYDSL_FP8_MAX_NUMEL) == ""
    assert sel.select_mla_prefill_backend(numel=1024) == sel.BACKEND_NAME


def test_requested_overrides_env(monkeypatch):
    monkeypatch.delenv(sel.OPT_IN_ENV, raising=False)
    monkeypatch.setattr(sel, "_on_gfx950", lambda: True)
    monkeypatch.setattr(sel, "_aiter_flydsl_fp8_available", lambda: True)
    assert sel.select_mla_prefill_backend(sel.BACKEND_NAME, numel=8) == sel.BACKEND_NAME
    assert sel.select_mla_prefill_backend("OTHER", numel=8) == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
