"""Self-contained CI entry point for the MXFP4 Engram layout tests.

CI environment preparation can fail while resolving the full sglang
 dependency tree (torch, triton, py-spy, pybase64, pydantic, quack-kernels,
...). The MXFP4 layout module ``python/sglang/srt/layers/engram_mxfp4.py`` is
stdlib-only, so its tests must be runnable with only pytest installed and
without importing the ``sglang`` package.

This module is deliberately self-contained: it loads the layout module and
the stdlib-only self-check by file path, never imports ``sglang``, and does
not rely on the repository root ``conftest.py``. It is the entry point CI
should call instead of running pytest from the repository root, which would
load the root configuration and trigger the full dependency install.

Run it with::

    PYTHONPATH=python python -m pytest -q \
        -c tests/layers/pytest.ini --rootdir=tests/layers \
        --confcutdir=tests/layers -p no:cacheprovider \
        tests/layers/test_engram_mxfp4_ci_entry.py

or, without pytest at all::

    PYTHONPATH=python python tests/layers/test_engram_mxfp4_ci_entry.py

or via the bundled runner::

    bash tests/layers/run_engram_mxfp4_tests.sh
"""

from __future__ import annotations

import importlib.util
import os
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")
if os.path.isdir(_PYTHON_DIR) and _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

_LAYOUT_PATH = os.path.join(
    _PYTHON_DIR, "sglang", "srt", "layers", "engram_mxfp4.py"
)
_SELFCHECK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_engram_mxfp4_selfcheck.py",
)


def _load_by_path(name: str, path: str):
    """Load a stdlib-only module by file path without importing sglang."""
    if not os.path.isfile(path):
        raise ImportError(f"module not found at {path!r}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_layout = _load_by_path("engram_mxfp4_ci_entry", _LAYOUT_PATH)
_selfcheck = _load_by_path("engram_mxfp4_selfcheck_ci_entry", _SELFCHECK_PATH)

# Re-export the stdlib-only self-check cases so pytest collects them here.
for _name, _obj in list(vars(_selfcheck).items()):
    if _name.startswith("test_") and callable(_obj):
        globals()[_name] = _obj


def test_ci_entry_loads_layout_module() -> None:
    """The layout module must load without importing the sglang package."""
    assert hasattr(_layout, "EngramLayout")
    assert hasattr(_layout, "decode_e2m1_nibble")
    assert hasattr(_layout, "allocate_row_buffer")
    layout = _layout.EngramLayout.from_metadata(
        256, (17, 19), {"engram_embed_quant": "mxfp4"}
    )
    assert layout.row_bytes == 128
    assert layout.total_row_bytes() == 136


def test_ci_entry_exposes_selfcheck_cases() -> None:
    """Guard: the CI entry must expose the full self-check suite."""
    names = [
        name
        for name, obj in vars(_selfcheck).items()
        if name.startswith("test_") and callable(obj)
    ]
    assert len(names) >= 40, f"expected >= 40 self-check cases, got {len(names)}"
    assert "test_selfcheck_suite_is_collectable" in names


def test_ci_entry_runs_without_pytest() -> None:
    """Guard: the self-check must be runnable without pytest or torch."""
    assert hasattr(_selfcheck, "run_selfcheck")
    assert callable(_selfcheck.run_selfcheck)
    assert _selfcheck.run_selfcheck() == 0


def test_ci_entry_does_not_import_sglang() -> None:
    """Guard: this entry must never import the sglang package.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. This entry point exists so the MXFP4 layout tests stay
    runnable in that situation, so it must not pull in ``sglang`` (which would
    trigger the very dependency resolution it is meant to avoid).
    """
    assert "sglang" not in sys.modules


def test_ci_runner_uses_minimal_deps_entry() -> None:
    """Guard: the runner must delegate to the minimal-dependency CI entry.

    CI environment preparation can fail while resolving the full sglang
    dependency tree (py-spy, pybase64, pydantic, quack-kernels, ...). The
    runner must not run pytest directly from the repository root, because that
    loads the root configuration and can trigger the full dependency install.
    It must delegate to ``ci_run_engram_mxfp4.sh``, which installs only pytest
    with ``--no-deps`` and otherwise falls back to the stdlib self-check.
    """
    runner = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "run_engram_mxfp4_tests.sh",
    )
    assert os.path.isfile(runner), f"missing runner: {runner}"
    with open(runner, encoding="utf-8") as handle:
        text = handle.read()
    assert "ci_run_engram_mxfp4.sh" in text, (
        "runner must delegate to the minimal-dependency CI entry"
    )
    assert "pip install -e" not in text, (
        "runner must not install the sglang package"
    )


def test_ci_entry_script_installs_pytest_without_deps() -> None:
    """Guard: the CI entry must install pytest with ``--no-deps``.

    Without ``--no-deps`` pip resolves sglang's install_requires (py-spy,
    pybase64, pydantic, quack-kernels, ...), which is exactly the resolution
    that fails during CI environment preparation. The entry must also set
    PYTHONPATH before installing and must expose a stdlib fallback.
    """
    entry = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ci_run_engram_mxfp4.sh",
    )
    assert os.path.isfile(entry), f"missing CI entry: {entry}"
    with open(entry, encoding="utf-8") as handle:
        text = handle.read()
    assert "--no-deps" in text, "CI entry must install pytest with --no-deps"
    assert "PYTHONPATH" in text, "CI entry must set PYTHONPATH"
    assert "pip_status" in text, (
        "CI entry must capture the pip exit code explicitly"
    )
    assert "pip install -e" not in text, (
        "CI entry must not install the sglang package"
    )
    assert "pip install sglang" not in text, (
        "CI entry must not install the sglang package"
    )
    assert "run_stdlib_selfcheck" in text, (
        "CI entry must expose a stdlib self-check fallback"
    )
    assert "status=" in text, (
        "CI entry must log which path was taken"
    )
    assert "exit 0" in text, (
        "CI entry must exit 0 on the stdlib fallback path"
    )
    assert "--engram-mxfp4-stdlib-only" in text, (
        "CI entry must pass the stdlib-only marker to pytest"
    )
    assert "--confcutdir=tests/layers" in text, (
        "CI entry must skip the repository root conftest.py"
    )
    assert "--rootdir=tests/layers" in text, (
        "CI entry must pin rootdir so the root config is not loaded"
    )


def test_ci_entry_script_is_executable_and_self_contained() -> None:
    """Guard: the CI entry must be a runnable, self-contained script.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. The entry point must therefore be a standalone shell
    script that sets PYTHONPATH itself, installs only pytest with --no-deps,
    and falls back to the stdlib self-check when pytest is unavailable. This
    test asserts the script is present, non-empty, and contains the expected
    command chain so CI cannot silently fall back to `pip install -e python`.
    """
    entry = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ci_run_engram_mxfp4.sh",
    )
    assert os.path.isfile(entry), f"missing CI entry: {entry}"
    with open(entry, encoding="utf-8") as handle:
        text = handle.read()
    assert text.strip(), "CI entry must not be empty"
    assert text.startswith("#!/usr/bin/env bash"), (
        "CI entry must be a bash script"
    )
    assert "set -uo pipefail" in text, (
        "CI entry must fail fast on unset variables and pipe errors"
    )
    assert "set -euo pipefail" not in text, (
        "CI entry must not use set -e, which would skip the fallback"
    )
    assert "python -m pip install --no-deps" in text, (
        "CI entry must install pytest with --no-deps"
    )
    assert "python -m pytest" in text, (
        "CI entry must run pytest through the minimal config"
    )
    assert "python \"$SELFCHECK\"" in text, (
        "CI entry must run the stdlib self-check directly"
    )


def test_ci_entry_never_resolves_sglang_dependencies() -> None:
    """Guard: the CI entry must not resolve the sglang dependency tree.

    CI environment preparation fails while pip resolves sglang's
    install_requires (py-spy, pybase64, pydantic, quack-kernels, regex, ...).
    The entry must therefore never install the sglang package itself and must
    never run a bare ``pip install`` without ``--no-deps``, which would pull
    in that tree. This asserts on the actual command chain in the script, not
    just on the presence of a flag.
    """
    entry = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ci_run_engram_mxfp4.sh",
    )
    assert os.path.isfile(entry), f"missing CI entry: {entry}"
    with open(entry, encoding="utf-8") as handle:
        text = handle.read()

    # No install of the sglang package itself, in any spelling.
    for forbidden in (
        "pip install -e",
        "pip install sglang",
        "pip install .",
        "pip install -e .",
        "pip install -e python",
    ):
        assert forbidden not in text, (
            f"CI entry must not run {forbidden!r} (resolves sglang deps)"
        )

    # Every pip install line must carry --no-deps so sglang's install_requires
    # is never resolved.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "pip install" in stripped:
            assert "--no-deps" in stripped, (
                f"pip install without --no-deps: {stripped!r}"
            )

    # The stdlib fallback must be reachable and return success.
    assert "run_stdlib_selfcheck" in text
    assert "exit 0" in text


def test_runner_delegates_to_minimal_deps_entry() -> None:
    """Guard: the runner must delegate to the minimal-dependency entry.

    CI must not run pytest from the repository root, because that loads the
    root configuration and can trigger the full sglang dependency install.
    The runner must delegate to ``ci_run_engram_mxfp4.sh`` and must not
    install the sglang package itself.
    """
    runner = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "run_engram_mxfp4_tests.sh",
    )
    assert os.path.isfile(runner), f"missing runner: {runner}"
    with open(runner, encoding="utf-8") as handle:
        text = handle.read()
    assert "ci_run_engram_mxfp4.sh" in text, (
        "runner must delegate to the minimal-dependency CI entry"
    )
    assert "pip install -e" not in text, (
        "runner must not install the sglang package"
    )
    assert "pip install sglang" not in text, (
        "runner must not install the sglang package"
    )
    # The runner must not invoke pytest directly from the repository root.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "python -m pytest" not in stripped, (
            f"runner must not run pytest directly: {stripped!r}"
        )


if __name__ == "__main__":
    raise SystemExit(_selfcheck.run_selfcheck())
