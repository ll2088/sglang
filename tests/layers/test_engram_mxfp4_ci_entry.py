"""Guard tests for the dependency-free CI entry points of the MXFP4 tests.

CI environment preparation can fail while resolving the full sglang
 dependency tree (py-spy, pybase64, pydantic, pyzmq, quack-kernels, regex,
...). The MXFP4 layout module ``python/sglang/srt/layers/engram_mxfp4.py`` is
stdlib-only, so its tests must be runnable without installing the sglang
package.

These tests assert that the CI entry scripts keep the hard guard against
``pip install -e`` / ``pip install sglang`` at the very top of the file and
delegate to the pure-stdlib entry point, so a future edit cannot silently
reintroduce the failing dependency resolution. They are stdlib-only and never
import ``sglang``.

Run with::

    PYTHONPATH=python python -m pytest -q --confcutdir=tests/layers \
        -p no:cacheprovider tests/layers/test_engram_mxfp4_ci_entry.py
"""

from __future__ import annotations

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

_RUNNER = os.path.join(_THIS_DIR, "run_engram_mxfp4_tests.sh")
_INSTALL = os.path.join(_THIS_DIR, "ci_install_minimal_deps.sh")
_STDLIB = os.path.join(_THIS_DIR, "run_engram_mxfp4_stdlib.sh")

# The guard must appear within the first lines of the script, before any
# command that could trigger dependency resolution.
_GUARD_WINDOW = 40


def _read(path: str) -> str:
    assert os.path.isfile(path), f"missing script: {path}"
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _guard_line(text: str) -> int:
    """Return the 1-based line of the first pip-install guard, or -1."""
    for index, line in enumerate(text.splitlines(), start=1):
        if "pip" in line and "install" in line and "grep" in line:
            return index
    return -1


def test_runner_guard_is_at_the_top() -> None:
    """The runner must reject pip installs before doing anything else."""
    text = _read(_RUNNER)
    line = _guard_line(text)
    assert line != -1, "run_engram_mxfp4_tests.sh has no pip-install guard"
    assert line <= _GUARD_WINDOW, (
        f"pip-install guard must be within the first {_GUARD_WINDOW} lines, "
        f"found at line {line}"
    )


def test_install_script_guard_is_at_the_top() -> None:
    """The minimal-deps script must reject pip installs before anything else."""
    text = _read(_INSTALL)
    line = _guard_line(text)
    assert line != -1, "ci_install_minimal_deps.sh has no pip-install guard"
    assert line <= _GUARD_WINDOW, (
        f"pip-install guard must be within the first {_GUARD_WINDOW} lines, "
        f"found at line {line}"
    )


def test_runner_delegates_to_stdlib_entry() -> None:
    """The runner must delegate to the pure-stdlib entry point."""
    text = _read(_RUNNER)
    assert "run_engram_mxfp4_stdlib.sh" in text
    assert "exec bash" in text


def test_install_script_delegates_to_stdlib_entry() -> None:
    """The minimal-deps script must delegate to the pure-stdlib entry point."""
    text = _read(_INSTALL)
    assert "run_engram_mxfp4_stdlib.sh" in text
    assert "exec bash" in text


def test_stdlib_entry_exists_and_is_dependency_free() -> None:
    """The stdlib entry point must exist and never install anything."""
    text = _read(_STDLIB)
    assert text.startswith("#!/usr/bin/env bash")
    assert "pip install" not in text, "stdlib entry must not install anything"
    assert "test_engram_mxfp4_selfcheck.py" in text
    assert "status=passed" in text
    assert "exit 0" in text


def test_scripts_do_not_install_sglang() -> None:
    """None of the CI entry scripts may install the sglang package."""
    for path in (_RUNNER, _INSTALL, _STDLIB):
        text = _read(path)
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "pip install sglang" not in stripped, (
                f"{os.path.basename(path)} must not install sglang: {stripped!r}"
            )
            assert "pip install -e" not in stripped, (
                f"{os.path.basename(path)} must not run pip install -e: {stripped!r}"
            )
