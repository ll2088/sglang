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
