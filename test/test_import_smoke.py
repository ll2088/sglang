"""Lightweight import smoke tests for the sglang package.

These tests only exercise the import chain (no model loading, no GPU work) so
that a broken import is reported in seconds and points at the first failing
submodule instead of surfacing as a slow integration-test failure.
"""

import importlib
import os
import re
import subprocess
import sys
import textwrap

import pytest

# Keep the child interpreter budget small: the whole point of this smoke test
# is to surface a broken import in seconds, not to wait on a hung import.
_CHILD_IMPORT_TIMEOUT_S = 20


def _first_failing_module(exc: BaseException) -> str:
    """Best-effort extraction of the first failing module name from a traceback.

    Walks the traceback from the outermost frame inward and returns the first
    frame that belongs to the ``sglang`` package, so the reported name points
    at the submodule that actually broke rather than at the test harness.
    Falls back to the innermost known module name when no sglang frame exists.
    """
    tb = exc.__traceback__
    first_sglang: str | None = None
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


_FAILING_MODULE_RE = re.compile(r'File "[^"]*[\\/]([^\\/]+)\.py", line \d+')


def _first_failing_module_from_stderr(stderr: str) -> str:
    """Extract the first sglang module file mentioned in a subprocess traceback.

    Prefers the first ``sglang`` frame so the message names the submodule that
    broke; falls back to the last file mentioned when no sglang frame is found.
    """
    matches = _FAILING_MODULE_RE.findall(stderr or "")
    if not matches:
        return "<unknown>"
    for name in matches:
        if name == "sglang" or name.startswith("sglang"):
            return name
    return matches[-1]


def test_import_sglang_in_fresh_interpreter():
    """A fresh interpreter must import sglang without preloaded state."""
    code = textwrap.dedent(
        """
        import sglang
        import sglang._platform_stubs
        print(sglang.__version__)
        """
    )
    # Inherit the parent's environment (including PYTHONPATH) so the child
    # resolves exactly the same sglang package as the parent test process.
    # Dropping PYTHONPATH would make the child fall back to a different
    # install and mask the very breakage this test is meant to catch.
    env = dict(os.environ)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=_CHILD_IMPORT_TIMEOUT_S,
            env=env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"fresh-interpreter import did not finish within "
            f"{_CHILD_IMPORT_TIMEOUT_S}s",
            pytrace=False,
        )
    if proc.returncode != 0:
        failing = _first_failing_module_from_stderr(proc.stderr)
        pytest.fail(
            f"fresh-interpreter import failed in module {failing!r}:\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            pytrace=False,
        )
    assert proc.stdout.strip(), "fresh interpreter produced no version output"


def test_import_smoke_reports_first_failing_module():
    """The failure helper must name the first broken sglang submodule."""
    try:
        importlib.import_module("sglang._nonexistent_smoke_probe")
    except Exception as exc:  # noqa: BLE001 - exercising the reporting helper
        failing = _first_failing_module(exc)
    else:  # pragma: no cover - the probe module must not exist
        pytest.fail("probe module unexpectedly importable")

    assert failing == "sglang" or failing.startswith("sglang."), failing


def test_import_smoke_reports_first_failing_module_from_stderr():
    """The subprocess helper must name the first sglang frame in a traceback."""
    stderr = textwrap.dedent(
        """
        Traceback (most recent call last):
          File "/repo/python/sglang/__init__.py", line 10, in <module>
            from sglang.srt.environ import redirect_third_party_caches
          File "/repo/python/sglang/srt/environ.py", line 3, in <module>
            import torch
        ModuleNotFoundError: No module named 'torch'
        """
    )
    assert _first_failing_module_from_stderr(stderr) == "sglang"

    assert _first_failing_module_from_stderr("") == "<unknown>"


def test_import_smoke_reports_non_sglang_frame_fallback():
    """Without any sglang frame the helper falls back to the last file seen."""
    stderr = textwrap.dedent(
        """
        Traceback (most recent call last):
          File "/usr/lib/python3.14/site-packages/torch/__init__.py", line 1, in <module>
            import _C
        ImportError: cannot import name '_C'
        """
    )
    assert _first_failing_module_from_stderr(stderr) == "__init__"
