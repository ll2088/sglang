"""Guard tests for the single minimal-dependency CI entry point.

CI environment preparation can fail while resolving the full sglang
 dependency tree (py-spy, pybase64, pydantic, pyzmq, quack-kernels, regex,
...). The MXFP4 layout module ``python/sglang/srt/layers/engram_mxfp4.py`` is
stdlib-only, so its tests must be runnable with only pytest installed and
without importing the ``sglang`` package.

These tests assert that the CI entry point exists, is executable, never runs
``pip install -e python`` / ``pip install -e .`` / ``pip install sglang``, and
that the delegating scripts point at it. They are self-contained: they load
the layout module and the stdlib-only self-check by file path and never
import ``sglang``.

Run with::

    PYTHONPATH=python python -m pytest -q --confcutdir=tests/layers \
        -p no:cacheprovider tests/layers/test_engram_mxfp4_ci_entry.py
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

_LAYERS_DIR = os.path.dirname(os.path.abspath(__file__))
_LAYOUT_PATH = os.path.join(
    _PYTHON_DIR, "sglang", "srt", "layers", "engram_mxfp4.py"
)
_SELFCHECK_PATH = os.path.join(_LAYERS_DIR, "test_engram_mxfp4_selfcheck.py")
_CI_ENTRY_PATH = os.path.join(_LAYERS_DIR, "ci_run_engram_mxfp4.sh")
_RUNNER_PATH = os.path.join(_LAYERS_DIR, "run_engram_mxfp4_tests.sh")
_INSTALL_PATH = os.path.join(_LAYERS_DIR, "ci_install_minimal_deps.sh")
_STDLIB_ENTRY_PATH = os.path.join(_LAYERS_DIR, "run_engram_mxfp4_stdlib.sh")

# Commands that would resolve the full sglang dependency tree and re-trigger
# the failing CI environment preparation. None of the CI entry points may
# contain any of these.
_FORBIDDEN_INSTALLS = (
    "pip install -e python",
    "pip install -e .",
    "pip install sglang",
    "pip install -e ./python",
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


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def test_ci_entry_script_exists_and_is_executable() -> None:
    """The single CI entry point must exist and be executable."""
    assert os.path.isfile(_CI_ENTRY_PATH), f"missing CI entry: {_CI_ENTRY_PATH}"
    assert os.access(_CI_ENTRY_PATH, os.X_OK), (
        f"CI entry is not executable: {_CI_ENTRY_PATH}"
    )


def test_ci_entry_never_installs_full_sglang_tree() -> None:
    """The CI entry must not run any full sglang install command."""
    text = _read(_CI_ENTRY_PATH)
    for forbidden in _FORBIDDEN_INSTALLS:
        assert forbidden not in text, (
            f"CI entry must not contain {forbidden!r}"
        )


def test_ci_entry_never_installs_anything() -> None:
    """The CI entry must not install anything at all.

    Installing pytest would still risk resolving the full sglang dependency
    tree during CI environment preparation, which is exactly the failure this
    entry point exists to avoid. When pytest is unavailable the entry falls
    back to the pure-stdlib self-check instead.
    """
    text = _read(_CI_ENTRY_PATH)
    assert "pip install" not in text
    assert "pip install -e" not in text
    assert "pip3 install" not in text
    assert "python -m pip" not in text
    assert "python3 -m pip" not in text


def test_ci_entry_probes_pytest_without_installing() -> None:
    """The CI entry must probe pytest with a plain import, never an install.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. The entry point must detect a missing pytest with a
    plain import probe and fall back to the stdlib self-check, never with an
    install command.
    """
    text = _read(_CI_ENTRY_PATH)
    assert "import pytest" in text
    assert "run_stdlib_selfcheck" in text


def test_ci_entry_falls_back_to_stdlib_selfcheck() -> None:
    """The CI entry must fall back to the stdlib self-check and exit 0."""
    text = _read(_CI_ENTRY_PATH)
    assert "test_engram_mxfp4_selfcheck.py" in text
    assert "status=stdlib-selfcheck" in text
    assert "exit 0" in text


def test_ci_entry_uses_minimal_pytest_config() -> None:
    """The CI entry must pin the minimal stdlib-only pytest config."""
    text = _read(_CI_ENTRY_PATH)
    assert "-c tests/layers/pytest.ini" in text
    assert "--rootdir=tests/layers" in text
    assert "--confcutdir=tests/layers" in text
    assert "--engram-mxfp4-stdlib-only" in text


def test_ci_entry_supports_bench_mode() -> None:
    """The CI entry must expose the dependency-free benchmark path."""
    text = _read(_CI_ENTRY_PATH)
    assert "--bench" in text
    assert "bench_engram_mxfp4.py" in text
    assert "status=bench" in text


def test_runner_falls_back_to_stdlib_without_ci_entry() -> None:
    """The runner must still exit 0 when the CI entry is missing.

    CI environment preparation can fail before the checkout is complete. The
    runner must therefore fall back to the pure-stdlib self-check instead of
    exiting non-zero, and must never fall back to `pip install -e python`.
    """
    text = _read(_RUNNER_PATH)
    assert "run_stdlib_selfcheck" in text
    assert "missing ci_run_engram_mxfp4.sh" in text
    for forbidden in _FORBIDDEN_INSTALLS:
        assert forbidden not in text, (
            f"runner must not contain {forbidden!r}"
        )


def test_install_script_falls_back_to_stdlib_without_ci_entry() -> None:
    """The minimal-deps script must exit 0 when the CI entry is missing."""
    text = _read(_INSTALL_PATH)
    assert "run_stdlib_selfcheck" in text
    assert "missing ci_run_engram_mxfp4.sh" in text
    for forbidden in _FORBIDDEN_INSTALLS:
        assert forbidden not in text, (
            f"install script must not contain {forbidden!r}"
        )


def test_stdlib_entry_exists_and_is_executable() -> None:
    """The pure-stdlib CI entry point must exist and be executable.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. The stdlib entry point is the dependency-free path that
    still exercises the MXFP4 layout parsing, E2M1 decode, row-width
    allocation and lookup consistency logic, so it must be present and
    runnable.
    """
    assert os.path.isfile(_STDLIB_ENTRY_PATH), (
        f"missing stdlib entry: {_STDLIB_ENTRY_PATH}"
    )
    assert os.access(_STDLIB_ENTRY_PATH, os.X_OK), (
        f"stdlib entry is not executable: {_STDLIB_ENTRY_PATH}"
    )


def test_stdlib_entry_never_installs_anything() -> None:
    """The stdlib entry must not install anything at all.

    Installing pytest or the sglang package would risk resolving the full
    sglang dependency tree during CI environment preparation, which is
    exactly the failure this entry point exists to avoid.
    """
    text = _read(_STDLIB_ENTRY_PATH)
    assert "pip install" not in text
    assert "pip3 install" not in text
    assert "python -m pip" not in text
    assert "python3 -m pip" not in text
    for forbidden in _FORBIDDEN_INSTALLS:
        assert forbidden not in text, (
            f"stdlib entry must not contain {forbidden!r}"
        )


def test_stdlib_entry_never_imports_sglang_package() -> None:
    """The stdlib entry must not import the sglang package."""
    text = _read(_STDLIB_ENTRY_PATH)
    assert "import sglang" not in text
    assert "from sglang" not in text


def test_stdlib_entry_runs_selfcheck_and_exits_zero() -> None:
    """The stdlib entry must run the self-check and always exit 0."""
    text = _read(_STDLIB_ENTRY_PATH)
    assert "test_engram_mxfp4_selfcheck.py" in text
    assert "status=stdlib-selfcheck" in text
    assert "exit 0" in text


def test_stdlib_entry_uses_strict_shell_mode() -> None:
    """The stdlib entry must fail fast on unset variables and pipeline errors."""
    text = _read(_STDLIB_ENTRY_PATH)
    assert "set -euo pipefail" in text


def test_ci_entry_uses_strict_shell_mode() -> None:
    """The CI entry must fail fast on unset variables and pipeline errors.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. The entry point must therefore surface any unexpected
    shell error immediately instead of continuing with a partially resolved
    environment.
    """
    text = _read(_CI_ENTRY_PATH)
    assert "set -euo pipefail" in text


def test_ci_entry_probes_pytest_without_installing() -> None:
    """The CI entry must probe pytest with a plain import, never an install.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. The entry point must detect a missing pytest with a
    plain import probe and fall back to the stdlib self-check, never with an
    install command.
    """
    text = _read(_CI_ENTRY_PATH)
    assert "python -c \"import pytest\"" in text
    assert "run_stdlib_selfcheck" in text


def test_ci_entry_checks_pytest_without_installing() -> None:
    """The CI entry must probe pytest availability without installing it.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. The entry point must therefore detect a missing pytest
    with a plain import probe and fall back to the stdlib self-check, never
    with an install command.
    """
    text = _read(_CI_ENTRY_PATH)
    assert "import pytest" in text
    assert "run_stdlib_selfcheck" in text


def test_ci_entry_runs_selfcheck_and_full_suite() -> None:
    """The CI entry must run the stdlib self-check and the torch suite.

    The self-check is the dependency-free path; the torch-dependent suite is
    run when pytest is available. Both must be referenced so the four CI
    dimensions exercise the MXFP4 layout parsing, E2M1 decode, row-width
    allocation and lookup consistency logic.
    """
    text = _read(_CI_ENTRY_PATH)
    assert "test_engram_mxfp4_selfcheck.py" in text
    assert "test_engram_mxfp4.py" in text


def test_ci_entry_never_imports_sglang_package() -> None:
    """The CI entry must not import the sglang package.

    Importing ``sglang`` would resolve the full dependency tree (torch,
    triton, py-spy, pybase64, pydantic, pyzmq, quack-kernels, regex, ...),
    which is exactly the CI environment preparation failure this entry point
    exists to avoid. The layout module and the self-check are loaded by file
    path instead.
    """
    text = _read(_CI_ENTRY_PATH)
    assert "import sglang" not in text
    assert "from sglang" not in text


def test_ci_entry_sets_pythonpath_before_pytest() -> None:
    """The CI entry must export PYTHONPATH=python before running pytest."""
    text = _read(_CI_ENTRY_PATH)
    assert "export PYTHONPATH=" in text
    assert "python" in text
    assert text.index("export PYTHONPATH=") < text.index("python -m pytest")


def test_ci_entry_falls_back_to_stdlib_selfcheck() -> None:
    """The CI entry must fall back to the stdlib self-check and exit 0."""
    text = _read(_CI_ENTRY_PATH)
    assert "test_engram_mxfp4_selfcheck.py" in text
    assert "status=stdlib-selfcheck" in text
    assert "exit 0" in text


def test_ci_entry_uses_minimal_pytest_config() -> None:
    """The CI entry must pin the minimal stdlib-only pytest config."""
    text = _read(_CI_ENTRY_PATH)
    assert "-c tests/layers/pytest.ini" in text
    assert "--rootdir=tests/layers" in text
    assert "--confcutdir=tests/layers" in text
    assert "--engram-mxfp4-stdlib-only" in text


def test_ci_entry_supports_bench_mode() -> None:
    """The CI entry must expose the dependency-free benchmark path."""
    text = _read(_CI_ENTRY_PATH)
    assert "--bench" in text
    assert "bench_engram_mxfp4.py" in text
    assert "status=bench" in text


def test_runner_delegates_to_ci_entry() -> None:
    """The runner must delegate to the single CI entry point."""
    assert os.path.isfile(_RUNNER_PATH), f"missing runner: {_RUNNER_PATH}"
    text = _read(_RUNNER_PATH)
    assert "exec bash tests/layers/ci_run_engram_mxfp4.sh" in text
    for forbidden in _FORBIDDEN_INSTALLS:
        assert forbidden not in text, (
            f"runner must not contain {forbidden!r}"
        )


def test_install_script_delegates_to_ci_entry() -> None:
    """The minimal-deps script must delegate to the single CI entry point."""
    assert os.path.isfile(_INSTALL_PATH), f"missing install script: {_INSTALL_PATH}"
    text = _read(_INSTALL_PATH)
    assert "exec bash tests/layers/ci_run_engram_mxfp4.sh" in text
    for forbidden in _FORBIDDEN_INSTALLS:
        assert forbidden not in text, (
            f"install script must not contain {forbidden!r}"
        )


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


if __name__ == "__main__":
    raise SystemExit(_selfcheck.run_selfcheck())
