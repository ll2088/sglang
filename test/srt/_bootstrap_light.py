"""Lightweight test bootstrap for the protobuf / streaming / queue tests.

These tests only exercise pure-Python helpers, so they must not require a
full ``pip install -e python`` (which pulls torch, CUDA wheels, etc.).  When
``sglang`` is importable we do nothing; otherwise we register minimal stub
modules for the heavy third-party imports that the target modules touch at
import time, then load the real source files by path.

Import this module *before* importing the modules under test::

    from test.srt import _bootstrap_light  # noqa: F401
    from sglang.srt.utils.protobuf_impl import warn_if_python_protobuf
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "python"


def _ensure_package(name: str, path: Path | None = None) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is not None:
        return module
    module = types.ModuleType(name)
    module.__path__ = [str(path)] if path is not None else []
    sys.modules[name] = module
    return module


def _stub(name: str, **attrs) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_by_path(module_name: str, file_path: Path) -> types.ModuleType:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_stubs() -> None:
    # torch is imported at module scope by several sglang modules; a stub with
    # the few attributes touched at import time is enough for our helpers.
    if importlib.util.find_spec("torch") is None:
        torch = _stub("torch")
        torch.Tensor = type("Tensor", (), {})
        torch.float32 = "float32"
        torch.cuda = _stub("torch.cuda", Event=type("Event", (), {}))
        torch.profiler = _stub(
            "torch.profiler",
            record_function=lambda *_a, **_k: (lambda fn: fn),
        )
        torch.get_device_module = lambda *_a, **_k: torch.cuda

    if importlib.util.find_spec("msgspec") is None:
        msgspec = _stub("msgspec")
        msgspec.Struct = type("Struct", (), {})
        msgspec.msgpack = _stub(
            "msgspec.msgpack",
            encode=lambda *_a, **_k: b"",
            decode=lambda *_a, **_k: None,
        )
        msgspec.structs = _stub("msgspec.structs", asdict=lambda obj: dict(obj))

    if importlib.util.find_spec("psutil") is None:
        _stub("psutil", Process=lambda: types.SimpleNamespace(create_time=lambda: 0.0))

    # google.protobuf is optional for the warning helper; when absent the
    # helper must silently skip, which is exactly what we want to test.


def _install_sglang_packages() -> None:
    if _SRC_ROOT.is_dir() and str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))
    _ensure_package("sglang", _SRC_ROOT / "sglang")
    _ensure_package("sglang.srt", _SRC_ROOT / "sglang" / "srt")
    _ensure_package("sglang.srt.utils", _SRC_ROOT / "sglang" / "srt" / "utils")


def _load_target_modules() -> None:
    utils_dir = _SRC_ROOT / "sglang" / "srt" / "utils"
    for name in ("protobuf_impl", "streaming", "async_queue"):
        path = utils_dir / f"{name}.py"
        if path.is_file():
            _load_by_path(f"sglang.srt.utils.{name}", path)


def bootstrap() -> None:
    """Idempotent: safe to call from every test module."""
    if getattr(bootstrap, "_done", False):
        return
    _install_stubs()
    _install_sglang_packages()
    _load_target_modules()
    bootstrap._done = True


bootstrap()
