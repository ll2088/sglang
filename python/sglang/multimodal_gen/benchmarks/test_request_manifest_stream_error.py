"""Tests that request manifests reject stream error records."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_request_manifest():
    module_path = Path(__file__).with_name("request_manifest.py")
    spec = importlib.util.spec_from_file_location("request_manifest", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


request_manifest = _load_request_manifest()


def _write_manifest(tmp_path: Path, records) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_manifest_with_stream_error_is_rejected(tmp_path):
    path = _write_manifest(
        tmp_path,
        [
            {"prompt": "hello"},
            {"prompt": "world", "error": {"message": "boom", "code": 500}},
        ],
    )
    with pytest.raises(ValueError, match="stream error"):
        request_manifest.load_request_manifest(path)


def test_manifest_without_error_is_loaded(tmp_path):
    path = _write_manifest(
        tmp_path,
        [
            {"prompt": "hello", "request_id": "r1"},
            {"prompt": "world", "request_id": "r2"},
        ],
    )
    loaded = request_manifest.load_request_manifest(path)
    assert [r.request_id for r in loaded.requests] == ["r1", "r2"]
