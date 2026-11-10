import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# These tests only exercise pure-Python capability probing / fallback logic and
# must stay collectable without a GPU or the full sglang install.


def _import_backend():
    try:
        from sglang.kernels.ops.attention import mla_flydsl_fp8_prefill as mod
    except Exception:  # noqa: BLE001
        return None
    return mod


_mod = _import_backend()

pytestmark = pytest.mark.skipif(
    _mod is None, reason="mla_flydsl_fp8_prefill backend not available"
)


def test_supports_compute_capability_requires_gfx950():
    backend_cls = _mod.AiterFlyDSLFP8PrefillBackend
    capability = MagicMock()
    with patch.object(_mod, "_is_rocm", return_value=False):
        assert not backend_cls.supports_compute_capability(capability)
    for on_gfx950 in (False, True):
        with (
            patch.object(_mod, "_is_rocm", return_value=True),
            patch.object(_mod, "_on_gfx950", return_value=on_gfx950),
        ):
            assert backend_cls.supports_compute_capability(capability) is on_gfx950


def test_is_available_requires_flydsl_fp8_attention():
    backend_cls = _mod.AiterFlyDSLFP8PrefillBackend
    flydsl = SimpleNamespace(
        flydsl_flash_attn_fp8_func=MagicMock(),
        flydsl_flash_attn_fp8_supported=MagicMock(),
    )
    with (
        patch.object(_mod, "_aiter_enabled", return_value=False),
        patch.dict(sys.modules, {"aiter.ops.flydsl": flydsl}),
    ):
        assert not backend_cls.is_available()
    with patch.object(_mod, "_aiter_enabled", return_value=True):
        with patch.dict(sys.modules, {"aiter.ops.flydsl": flydsl}):
            assert backend_cls.is_available()
        # AITER builds without the capability check are not supported.
        with patch.dict(sys.modules, {"aiter.ops.flydsl": SimpleNamespace()}):
            assert not backend_cls.is_available()


def test_validate_configuration_rejects_unsupported_head_dims():
    backend_cls = _mod.AiterFlyDSLFP8PrefillBackend
    dims = _mod.MLADimensions(
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
    )
    selector_config = _mod.MLAPrefillSelectorConfig(
        dtype=_mod.torch.bfloat16, mla_dimensions=dims
    )
    capability = MagicMock()
    with (
        patch.object(_mod, "_is_rocm", return_value=True),
        patch.object(_mod, "_on_gfx950", return_value=True),
        patch.object(_mod, "_aiter_enabled", return_value=True),
        patch.dict(
            sys.modules,
            {
                "aiter.ops.flydsl": SimpleNamespace(
                    flydsl_flash_attn_fp8_func=MagicMock(),
                    flydsl_flash_attn_fp8_supported=lambda *a, **k: False,
                )
            },
        ),
    ):
        reasons = backend_cls.validate_configuration(capability, selector_config)
        assert reasons


def test_fallback_to_parent_when_kernel_cannot_take_input():
    """Empty or oversized inputs must fall back to the parent BF16 path."""
    backend_cls = _mod.AiterFlyDSLFP8PrefillBackend
    assert hasattr(backend_cls, "_should_fallback_to_parent")
    assert backend_cls._should_fallback_to_parent(0)
    assert backend_cls._should_fallback_to_parent(2**31)
    assert not backend_cls._should_fallback_to_parent(1024)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
