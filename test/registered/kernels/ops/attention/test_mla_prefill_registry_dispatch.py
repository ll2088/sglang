import sys

import pytest

# Registry dispatch tests are pure metadata checks and must not require a GPU.


def _import_registry():
    try:
        from sglang.kernels import registry as reg
    except Exception:  # noqa: BLE001
        return None
    return reg


_reg = _import_registry()

pytestmark = pytest.mark.skipif(
    _reg is None, reason="sglang.kernels.registry not available"
)


def test_registry_has_mla_prefill_ops():
    ops = _reg.registry.ops()
    assert any("mla" in op for op in ops), ops


def test_opt_in_backend_registered_but_not_default():
    """The FlyDSL FP8 backend must be registered but never auto-selected."""
    specs = [
        s
        for s in _reg.registry.all_specs()
        if "flydsl" in s.target.lower() or "flydsl" in s.op.lower()
    ]
    # If the backend is not present in this tree, the test is a no-op.
    if not specs:
        pytest.skip("no FlyDSL FP8 prefill spec registered")
    for spec in specs:
        assert not getattr(spec, "auto_select", False), spec


def test_registry_rejects_conflicting_registration():
    from sglang.kernels.spec import KernelBackend, KernelSpec

    op = "__test__.conflict"
    a = KernelSpec(op=op, backend=KernelBackend.TRITON, target="a.b")
    b = KernelSpec(op=op, backend=KernelBackend.TRITON, target="c.d")
    _reg.registry.register(a)
    with pytest.raises(ValueError):
        _reg.registry.register(b)


def test_registry_idempotent_registration():
    from sglang.kernels.spec import KernelBackend, KernelSpec

    op = "__test__.idempotent"
    spec = KernelSpec(op=op, backend=KernelBackend.TRITON, target="a.b")
    _reg.registry.register(spec)
    _reg.registry.register(spec)
    assert len(_reg.registry.get(op)) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
