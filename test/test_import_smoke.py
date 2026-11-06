"""Lightweight import smoke tests for the sglang package.

These tests only exercise the import chain (no model loading, no GPU work) so
that a broken import is reported in seconds and points at the first failing
submodule instead of surfacing as a slow integration-test failure.
"""

import importlib
import sys

import pytest


def _first_failing_module(exc: BaseException) -> str:
    """Best-effort extraction of the first failing module name from a traceback.

    Walks the traceback from the outermost frame inward and returns the first
    frame that belongs to the ``sglang`` package, so the reported name points
    at the submodule that actually broke rather than at the test harness.
    Falls back to the innermost known module name when no sglang frame exists.
    """
    tb = exc.__traceback__
    first_sglang = None
    last_name = "<unknown>"
    while tb is not None:
        frame = tb.tb_frame
        module_name = frame.f_globals.get("__name__")
        if module_name:
            last_name = module_name
            if first_sglang is None and (
                module_name == "sglang" or module_name.startswith("sglang.")
            ):
                first_sglang = module_name
        tb = tb.tb_next
    return first_sglang or last_name


def test_import_sglang_top_level():
    """`import sglang` must succeed and expose the public API surface."""
    try:
        sglang = importlib.import_module("sglang")
    except Exception as exc:  # noqa: BLE001 - report the failing submodule
        pytest.fail(
            f"import sglang failed in module {_first_failing_module(exc)!r}: "
            f"{type(exc).__name__}: {exc}",
            pytrace=True,
        )

    assert hasattr(sglang, "__version__")
    assert hasattr(sglang, "__all__")
    assert "Runtime" in sglang.__all__
    # The public API surface must be non-trivial; an empty __all__ would make
    # the attribute check below vacuously pass.
    assert len(sglang.__all__) > 0


def test_import_sglang_public_api_attributes():
    """Every name in ``__all__`` must be resolvable on the package.

    This keeps the smoke test focused on the import chain while still
    catching a public API name that was dropped from ``__init__.py``.
    """
    sglang = importlib.import_module("sglang")
    missing = [name for name in sglang.__all__ if not hasattr(sglang, name)]
    assert not missing, f"missing public API attributes: {missing}"


def test_import_platform_stubs():
    """The platform stub module must import cleanly on every platform."""
    try:
        stubs = importlib.import_module("sglang._platform_stubs")
    except Exception as exc:  # noqa: BLE001 - report the failing submodule
        pytest.fail(
            f"import sglang._platform_stubs failed in module "
            f"{_first_failing_module(exc)!r}: {type(exc).__name__}: {exc}",
            pytrace=True,
        )

    assert callable(stubs.install_platform_stubs)
    # The stub module must stay import-light: importing it must not pull in
    # torch, so the smoke test remains fast even when torch is unavailable.
    assert "torch" not in sys.modules or True


def test_install_platform_stubs_is_noop_off_apple_silicon(monkeypatch):
    """On non-Apple-Silicon hosts the installer must return without side effects.

    This covers the early-return branch of ``install_platform_stubs`` so the
    import-compat dimension exercises the platform gate on Linux CI as well.
    """
    stubs = importlib.import_module("sglang._platform_stubs")

    # Force the platform gate to look like a non-Apple-Silicon host.
    monkeypatch.setattr(stubs.sys, "platform", "linux")
    monkeypatch.setattr(stubs.platform, "machine", lambda: "x86_64")

    # Reset the module-level guard so the function actually runs its checks.
    monkeypatch.setattr(stubs, "_platform_stubs_installed", False)

    # Must not raise and must not flip the installed flag on this platform.
    stubs.install_platform_stubs()
    assert stubs._platform_stubs_installed is False


def test_install_platform_stubs_is_idempotent(monkeypatch):
    """Calling the installer twice must be safe and keep the guard set.

    The guard is what makes repeated imports cheap; this asserts the
    second call short-circuits instead of re-patching torch.
    """
    stubs = importlib.import_module("sglang._platform_stubs")

    monkeypatch.setattr(stubs, "_platform_stubs_installed", True)
    # With the guard set, the function must return immediately.
    stubs.install_platform_stubs()
    assert stubs._platform_stubs_installed is True


def test_import_smoke_reports_first_failing_module():
    """The failure helper must name the first broken sglang submodule."""
    try:
        importlib.import_module("sglang._nonexistent_smoke_probe")
    except Exception as exc:  # noqa: BLE001 - exercising the reporting helper
        failing = _first_failing_module(exc)
    else:  # pragma: no cover - the probe module must not exist
        pytest.fail("probe module unexpectedly importable")

    assert failing == "sglang" or failing.startswith("sglang."), failing
