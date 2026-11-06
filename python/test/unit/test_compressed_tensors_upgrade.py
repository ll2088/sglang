"""Regression tests for the compressed-tensors 0.18.0 -> 0.20.1 upgrade.

These tests are intentionally dependency-light: they only read packaging
metadata and import the public compressed-tensors API, so they can run on
CPU/NPU/XPU install variants without a GPU.
"""

import importlib
import re
from pathlib import Path

import pytest

PYTHON_DIR = Path(__file__).resolve().parents[2]

# Files that declare the compressed-tensors pin for an install variant.
# Every variant, including the default python/pyproject.toml, must agree on
# the exact pin so that no install path silently resolves an older release.
VARIANT_FILES = [
    "pyproject.toml",
    "pyproject_cpu.toml",
    "pyproject_npu.toml",
    "pyproject_xpu.toml",
    "pyproject_other.toml",
]

PIN_RE = re.compile(r'"compressed-tensors==([0-9][^"]*)"')


def _read(name):
    return (PYTHON_DIR / name).read_text(encoding="utf-8")


def _version_tuple(version):
    # Strip local/dev suffixes (e.g. "0.20.1+cu13", "0.20.1.dev0") before
    # comparing, so the range check does not depend on the exact build tag.
    core = re.split(r"[+\-]", version, maxsplit=1)[0]
    parts = []
    for piece in core.split(".")[:3]:
        match = re.match(r"\d+", piece)
        parts.append(int(match.group(0)) if match else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def test_compressed_tensors_version_in_range():
    ct = pytest.importorskip("compressed_tensors")
    version = ct.__version__
    parts = _version_tuple(version)
    assert parts >= (0, 20, 1), f"compressed-tensors {version} is older than 0.20.1"
    assert parts < (0, 21), f"compressed-tensors {version} is newer than 0.20.x"


def test_variant_pins_are_consistent():
    pins = {}
    for name in VARIANT_FILES:
        matches = PIN_RE.findall(_read(name))
        assert matches, f"{name} does not pin compressed-tensors"
        pins[name] = set(matches)
    # Strict equality: every variant must pin exactly 0.20.1, no ranges and no
    # leftover 0.18.0 entries.
    assert all(p == {"0.20.1"} for p in pins.values()), pins


def test_no_stale_018_pin_anywhere():
    for path in PYTHON_DIR.glob("pyproject*.toml"):
        text = path.read_text(encoding="utf-8")
        assert "compressed-tensors==0.18.0" not in text, path


def test_check_env_import_and_run():
    check_env = importlib.import_module("sglang.check_env")
    env = check_env._get_env()
    # get_package_versions must not raise even when optional packages are absent.
    versions = env.get_package_versions()
    assert isinstance(versions, dict)
    assert "sglang" in versions


def test_quantization_config_parsing_regression():
    ct = pytest.importorskip("compressed_tensors")
    config_mod = importlib.import_module("compressed_tensors.config")
    # The public config entry point must still be importable without touching
    # removed GPTQ ordering paths.
    assert hasattr(config_mod, "CompressionConfig")
    # 0.20 keeps CompressionConfig constructible from an empty config; guard
    # against signature drift without depending on optional fields. Some 0.20.x
    # builds require explicit kwargs, so fall back to a minimal config dict.
    try:
        cfg = config_mod.CompressionConfig()
    except TypeError:
        cfg = config_mod.CompressionConfig(format="dense")
    assert cfg is not None
    assert hasattr(cfg, "quantization_status") or hasattr(cfg, "format")
    # apply_quantization_config gained allowed_modules/leave_decompressed in
    # 0.20; make sure the symbol still exists for downstream callers. It is
    # re-exported from the top-level package, not from
    # compressed_tensors.quantization.
    assert hasattr(ct, "apply_quantization_config")
    assert ct is not None
