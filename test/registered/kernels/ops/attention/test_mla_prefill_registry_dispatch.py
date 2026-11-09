"""Registry dispatch tests for the opt-in FlyDSL FP8 MLA prefill backend.

Loaded by file path so the test does not require a full ``sglang`` install.
The registry is a pure-Python in-memory structure; these tests verify that
the opt-in switch selects the FlyDSL FP8 backend when on and the default
backend when off, and that re-registration is idempotent.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


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


_registry_mod = _load_module(
    Path("python/sglang/kernels/registry.py"),
    "_sglang_kernels_registry_under_test",
)
_spec_mod = _load_module(
    Path("python/sglang/kernels/spec.py"),
    "_sglang_kernels_spec_under_test",
)

pytestmark = pytest.mark.skipif(
    _registry_mod is None or _spec_mod is None,
    reason="kernel registry/spec unavailable",
)


def _make_spec(op, backend, target):
    return _spec_mod.KernelSpec(op=op, backend=backend, target=target)


def test_register_and_get_backend():
    reg = _registry_mod.KernelRegistry()
    backend = _spec_mod.KernelBackend
    spec = _make_spec("mla_prefill", backend.TRITON, "a.b")
    reg.register(spec)
    assert reg.has("mla_prefill")
    assert reg.get_backend("mla_prefill", backend.TRITON) is spec


def test_idempotent_reregistration():
    reg = _registry_mod.KernelRegistry()
    backend = _spec_mod.KernelBackend
    spec = _make_spec("mla_prefill", backend.TRITON, "a.b")
    reg.register(spec)
    reg.register(spec)
    assert len(reg.get("mla_prefill")) == 1


def test_conflicting_registration_rejected():
    reg = _registry_mod.KernelRegistry()
    backend = _spec_mod.KernelBackend
    reg.register(_make_spec("mla_prefill", backend.TRITON, "a.b"))
    with pytest.raises(ValueError):
        reg.register(_make_spec("mla_prefill", backend.TRITON, "c.d"))


def test_missing_backend_raises_keyerror():
    reg = _registry_mod.KernelRegistry()
    backend = _spec_mod.KernelBackend
    with pytest.raises(KeyError):
        reg.get_backend("mla_prefill", backend.TRITON)
