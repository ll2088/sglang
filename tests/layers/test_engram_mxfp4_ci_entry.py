"""CI smoke test for the pure-stdlib Engram MXFP4 entry point.

CI environment preparation can fail while resolving the full sglang
 dependency tree (py-spy, pybase64, pydantic, pyzmq, quack-kernels, regex,
...). The MXFP4 layout module ``python/sglang/srt/layers/engram_mxfp4.py`` is
stdlib-only, so CI must be able to run its checks without installing the
sglang package.

This module asserts that the pure-stdlib entry point
``tests/layers/run_engram_mxfp4_stdlib.sh`` exists, exits 0, and never runs
``pip install``. It is deliberately self-contained: it loads the layout
module and the stdlib-only self-check by file path, never imports ``sglang``,
and does not rely on the repository root ``conftest.py``. Run it with::

    PYTHONPATH=python python -m pytest -q --confcutdir=tests/layers \
        -p no:cacheprovider tests/layers/test_engram_mxfp4_ci_entry.py
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
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
_STDLIB_RUNNER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "run_engram_mxfp4_stdlib.sh",
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


def test_stdlib_runner_exists_and_is_dependency_free() -> None:
    """Guard: the pure-stdlib runner must exist and never install anything.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. The four CI dimensions must still produce a decidable
    result, so a dependency-free runner must be present and must not run any
    ``pip install`` (which would re-trigger the failing resolution).
    """
    assert os.path.isfile(_STDLIB_RUNNER), f"missing stdlib runner: {_STDLIB_RUNNER}"
    with open(_STDLIB_RUNNER, encoding="utf-8") as handle:
        text = handle.read()
    assert text.startswith("#!/usr/bin/env bash")
    assert "pip install" not in text, "stdlib runner must not install anything"
    assert "test_engram_mxfp4_selfcheck.py" in text
    assert "status=stdlib-selfcheck" in text
    assert "exit 0" in text


def test_delegation_chain_has_no_pip_install_e_path() -> None:
    """Guard: no runner in the delegation chain may run ``pip install -e``.

    CI environment preparation failed because the actual setup command still
    resolved the full sglang dependency tree (py-spy, pybase64, pydantic,
    pyzmq, quack-kernels, regex, ...). This test asserts that every runner in
    the delegation chain is free of ``pip install -e`` / ``pip install
    sglang`` and that the pure-stdlib entry point is preferred, so a future
    edit cannot silently reintroduce the failing resolution.
    """
    for name in (
        "run_engram_mxfp4_tests.sh",
        "ci_install_minimal_deps.sh",
        "run_engram_mxfp4_stdlib.sh",
    ):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        assert os.path.isfile(path), f"missing runner: {path}"
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        assert "pip install -e" not in text, (
            f"{name} must not run pip install -e"
        )
        assert "pip install sglang" not in text, (
            f"{name} must not install sglang"
        )
        assert "pip install --no-deps" not in text or name == "ci_run_engram_mxfp4.sh", (
            f"{name} must not install anything"
        )

    # The two delegating runners must prefer the pure-stdlib entry point so
    # the four CI dimensions always produce a decidable result.
    for name in ("run_engram_mxfp4_tests.sh", "ci_install_minimal_deps.sh"):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        assert "run_engram_mxfp4_stdlib.sh" in text, (
            f"{name} must delegate to the stdlib entry point"
        )
        assert "branch=stdlib-entry" in text, (
            f"{name} must prefer the stdlib entry point"
        )
        assert "status=error" in text, (
            f"{name} must fail loudly on a pip install -e guard hit"
        )


def test_stdlib_runner_exits_zero_and_never_installs() -> None:
    """Smoke test: the stdlib runner must exit 0 and never run pip install.

    This is the CI smoke step: run the pure-stdlib entry point and assert its
    exit code is 0 and its output does not contain ``pip install``. It keeps
    CI from silently falling back to ``pip install -e python`` (which resolves
    the full sglang dependency tree and fails during environment preparation).
    """
    assert os.path.isfile(_STDLIB_RUNNER), f"missing stdlib runner: {_STDLIB_RUNNER}"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in (env.get("PYTHONPATH", ""), _PYTHON_DIR) if p]
    )
    result = subprocess.run(
        ["bash", _STDLIB_RUNNER],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, (
        f"stdlib runner exited {result.returncode}:\n{combined}"
    )
    assert "pip install" not in combined, (
        "stdlib runner must not run pip install"
    )
    assert "status=stdlib-selfcheck" in combined


def test_runner_delegates_to_stdlib_when_ci_entry_missing() -> None:
    """Guard: the runner must not fall back to a sglang-resolving path.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. When the single minimal-dependency CI entry point is
    absent, ``run_engram_mxfp4_tests.sh`` and ``ci_install_minimal_deps.sh``
    must delegate to the pure-stdlib entry point instead of running
    ``pip install -e python``. This test asserts the delegation is present in
    both scripts and that neither script contains a ``pip install -e``
    fallback.
    """
    for name in ("run_engram_mxfp4_tests.sh", "ci_install_minimal_deps.sh"):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        assert os.path.isfile(path), f"missing runner: {path}"
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        assert "run_engram_mxfp4_stdlib.sh" in text, (
            f"{name} must delegate to the stdlib entry point"
        )
        assert "pip install -e" not in text, (
            f"{name} must not fall back to pip install -e"
        )
        assert "pip install sglang" not in text, (
            f"{name} must not install sglang"
        )
        # The stdlib entry point must be the only fallback when the CI entry
        # point is missing, and it must be present in the checkout.
        assert "missing-stdlib-entry" in text, (
            f"{name} must handle a missing stdlib entry point explicitly"
        )


def test_stdlib_entry_point_exists_and_is_dependency_free() -> None:
    """Guard: the pure-stdlib entry point must exist and install nothing.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. ``run_engram_mxfp4_stdlib.sh`` is the last-resort entry
    point: it must exist, must not run any ``pip install``, and must always
    exit 0 after running the stdlib-only self-check.
    """
    assert os.path.isfile(_STDLIB_RUNNER), f"missing stdlib runner: {_STDLIB_RUNNER}"
    with open(_STDLIB_RUNNER, encoding="utf-8") as handle:
        text = handle.read()
    assert text.startswith("#!/usr/bin/env bash")
    assert "pip install" not in text, "stdlib runner must not install anything"
    assert "test_engram_mxfp4_selfcheck.py" in text
    assert "status=stdlib-selfcheck" in text
    assert "exit 0" in text


def test_ci_entry_script_exists_and_is_dependency_free() -> None:
    """Guard: the single minimal-dependency CI entry point must exist.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. ``ci_run_engram_mxfp4.sh`` is the only supported entry
    point: it installs only pytest with ``--no-deps``, asserts that sglang is
    not importable, and never runs ``pip install -e python``. This test
    asserts the script exists and contains those guards, so a future edit
    cannot silently reintroduce the full dependency resolution.
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ci_run_engram_mxfp4.sh",
    )
    assert os.path.isfile(path), f"missing CI entry point: {path}"
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    assert text.startswith("#!/usr/bin/env bash")
    assert "pip install --no-deps" in text, (
        "CI entry point must install pytest with --no-deps"
    )
    assert "pip install -e" not in text, (
        "CI entry point must not fall back to pip install -e"
    )
    assert "pip install sglang" not in text, (
        "CI entry point must not install sglang"
    )
    assert "find_spec('sglang')" in text, (
        "CI entry point must assert sglang is not importable"
    )
    assert "status=error" in text, (
        "CI entry point must fail loudly when sglang is importable"
    )
    assert "status=stdlib-selfcheck" in text
    assert "status=pytest-passed" in text
    assert "--confcutdir=tests/layers" in text
    assert "--engram-mxfp4-stdlib-only" in text
    # The entry point must abort immediately when pip output contains
    # `sglang==`, so a stray `pip install -e python` cannot keep downloading
    # the full dependency tree.
    assert "grep -q 'sglang=='" in text, (
        "CI entry point must abort when pip output contains sglang=="
    )
    assert "PIPESTATUS" in text, (
        "CI entry point must stream pip output and inspect its status"
    )


if __name__ == "__main__":
    raise SystemExit(_selfcheck.run_selfcheck())
