"""Lightweight import shims for tests that must run without full sglang deps.

Several unit tests only exercise small, dependency-free modules
(``sglang.srt.utils.protobuf_impl``, ``sglang.srt.utils.streaming``,
``sglang.srt.utils.async_queue``). Importing them through the real
``sglang`` package pulls in torch, transformers and the rest of the runtime,
which is not available in the lightweight CI job.

This module installs stub packages into ``sys.modules`` so that
``import sglang.srt.utils.<module>`` resolves to the real source file while
``sglang/__init__.py`` and ``sglang/srt/__init__.py`` are never executed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "python"


def _ensure_package(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is not None:
        return module
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _load_module(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install() -> None:
    """Install stub packages and preload the dependency-free modules."""
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

    _ensure_package("sglang", _SRC_ROOT / "sglang")
    _ensure_package("sglang.srt", _SRC_ROOT / "sglang" / "srt")
    _ensure_package("sglang.srt.utils", _SRC_ROOT / "sglang" / "srt" / "utils")

    utils_dir = _SRC_ROOT / "sglang" / "srt" / "utils"
    for module_name in ("protobuf_impl", "streaming", "async_queue"):
        _load_module(
            f"sglang.srt.utils.{module_name}",
            utils_dir / f"{module_name}.py",
        )
