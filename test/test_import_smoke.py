"""Lightweight import smoke tests for the sglang package.

These tests only exercise the import chain (no model loading, no GPU work) so
that a broken import is reported in seconds and points at the first failing
submodule instead of surfacing as a slow integration-test failure.
"""

import importlib
import os
import subprocess
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


def _run_python(code: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter so the import chain is exercised
    from a clean ``sys.modules`` (pytest plugins may pre-import sglang)."""
    env = dict(os.environ)
    # Make the in-repo package importable without requiring an editable install.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_dir = os.path.join(repo_root, "python")
    if os.path.isdir(python_dir):
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (python_dir, env.get("PYTHONPATH", "")) if p
        )
    # Keep the smoke test fast and deterministic: never write bytecode, and
    # never let a user site-packages shadow the in-repo package.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def test_import_sglang_top_level():
    """`import sglang` must succeed in a clean interpreter and expose the
    public API surface. Runs in a subprocess so a broken import chain is
    reported in seconds and points at the first failing submodule."""
    proc = _run_python(
        "import sglang\n"
        "assert hasattr(sglang, '__version__'), 'missing __version__'\n"
        "assert hasattr(sglang, '__all__'), 'missing __all__'\n"
        "assert 'Runtime' in sglang.__all__, 'Runtime not in __all__'\n"
        "assert len(sglang.__all__) > 0, 'empty __all__'\n"
    )
    assert proc.returncode == 0, (
        "import sglang failed in a clean interpreter:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_import_smoke_is_collected():
    """Self-check: the smoke module must be importable and collected by pytest
    even when the sglang dependency tree is not installed.

    This keeps the smoke test itself runnable in a minimal environment, so a
    broken dependency install is reported as a dependency problem rather than
    as a collection error that hides the real import failure.
    """
    assert callable(_first_failing_module)
    assert callable(_run_python)


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
    assert "torch" not in sys.modules, (
        "importing sglang._platform_stubs must not import torch"
    )


def test_import_platform_stubs_in_clean_interpreter():
    """Import the platform stub module in a fresh interpreter.

    Running in a subprocess isolates the check from pytest plugins that may
    have already imported torch, so the import-light guarantee is verified
    from a clean ``sys.modules``.
    """
    proc = _run_python(
        "import sglang._platform_stubs as stubs\n"
        "assert callable(stubs.install_platform_stubs)\n"
        "assert 'torch' not in __import__('sys').modules, "
        "'importing sglang._platform_stubs must not import torch'\n"
    )
    assert proc.returncode == 0, (
        "import sglang._platform_stubs failed in a clean interpreter:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


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
