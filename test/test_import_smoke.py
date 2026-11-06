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


def _first_failing_module(exc: BaseException) -> str:
    """Best-effort extraction of the first failing module name from a traceback."""
    tb = exc.__traceback__
    last_name = "<unknown>"
    while tb is not None:
        frame = tb.tb_frame
        module_name = frame.f_globals.get("__name__")
        if module_name:
            last_name = module_name
        tb = tb.tb_next
    return last_name


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
    """Extract the last module file mentioned in a subprocess traceback."""
    matches = _FAILING_MODULE_RE.findall(stderr or "")
    if not matches:
        return "<unknown>"
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
    # Drop PYTHONPATH so the child resolves the same installed package as the
    # parent test process; otherwise a stale path could mask a broken install.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    if proc.returncode != 0:
        failing = _first_failing_module_from_stderr(proc.stderr)
        pytest.fail(
            f"fresh-interpreter import failed in module {failing!r}:\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            pytrace=False,
        )
    assert proc.stdout.strip(), "fresh interpreter produced no version output"
