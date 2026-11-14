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
            assert "pip install ." not in stripped, (
                f"{os.path.basename(path)} must not run pip install .: {stripped!r}"
            )


def test_runner_delegates_to_stdlib_entry_only() -> None:
    """The runner must delegate to the stdlib entry and never to pytest.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. The runner must therefore exec the pure-stdlib entry
    point and must not contain a reachable pytest invocation that could
    resolve the sglang package.
    """
    text = _read(_RUNNER)
    assert "exec bash \"$STDLIB_ENTRY\"" in text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "pytest" not in stripped, (
            f"run_engram_mxfp4_tests.sh must not invoke pytest: {stripped!r}"
        )


def test_stdlib_entry_is_the_only_ci_path() -> None:
    """The stdlib entry must be the single CI path for these tests.

    This guards against a future edit that reintroduces a pytest-based CI
    path which resolves the full sglang dependency tree during environment
    preparation.
    """
    text = _read(_STDLIB)
    assert "test_engram_mxfp4_selfcheck.py" in text
    assert "status=passed" in text
    assert "exit 0" in text
    assert "pip install" not in text
    assert "pytest" not in text


def test_stdlib_entry_is_executable() -> None:
    """The stdlib entry must be executable so CI can invoke it directly.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. The stdlib entry is the single supported path, so it
    must carry the executable bit; otherwise a CI step that runs it directly
    (rather than via ``bash``) would fail before the self-check runs.
    """
    assert os.path.isfile(_STDLIB), f"missing stdlib entry: {_STDLIB}"
    assert os.access(_STDLIB, os.X_OK), (
        f"stdlib entry must be executable: {_STDLIB}"
    )


def test_ci_chain_reaches_stdlib_entry() -> None:
    """The CI call chain must actually reach the stdlib entry point.

    A previous regression added the stdlib entry but left CI on the old
    ``pip install -e python`` path, so the guard never ran. This test asserts
    that both the runner and the minimal-deps script delegate to the stdlib
    entry, so the dependency-free path is the one CI executes.
    """
    for path in (_RUNNER, _INSTALL):
        text = _read(path)
        assert "run_engram_mxfp4_stdlib.sh" in text, (
            f"{os.path.basename(path)} must delegate to the stdlib entry"
        )
        assert "exec bash" in text, (
            f"{os.path.basename(path)} must exec the stdlib entry"
        )


def test_stdlib_entry_does_not_import_sglang() -> None:
    """The stdlib entry must not import the sglang package.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. The stdlib entry must therefore run the self-check by
    file path and never trigger an ``import sglang`` that would resolve that
    tree.
    """
    text = _read(_STDLIB)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "import sglang" not in stripped, (
            f"stdlib entry must not import sglang: {stripped!r}"
        )
        assert "from sglang" not in stripped, (
            f"stdlib entry must not import from sglang: {stripped!r}"
        )
