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


if __name__ == "__main__":
    raise SystemExit(_selfcheck.run_selfcheck())
