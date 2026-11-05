"""Lightweight tests for request manifest stream error rejection."""

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
    / "request_manifest.py"
)


def _load_manifest_module():
    spec = importlib.util.spec_from_file_location("request_manifest_under_test", MANIFEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest_module():
    return _load_manifest_module()


def _write_manifest(tmp_path: Path, records) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_manifest_rejects_stream_error(manifest_module, tmp_path):
    path = _write_manifest(
        tmp_path,
        [
            {
                "prompt": "hello",
                "error": {"message": "boom", "code": 500},
            }
        ],
    )
    with pytest.raises(ValueError, match="stream error"):
        manifest_module.load_request_manifest(path)


def test_manifest_rejects_string_stream_error(manifest_module, tmp_path):
    path = _write_manifest(
        tmp_path,
        [
            {
                "prompt": "hello",
                "error": "internal server error during generation",
            }
        ],
    )
    with pytest.raises(ValueError, match="stream error"):
        manifest_module.load_request_manifest(path)


def test_manifest_accepts_null_error(manifest_module, tmp_path):
    path = _write_manifest(tmp_path, [{"prompt": "hello", "error": None}])
    loaded = manifest_module.load_request_manifest(path)
    assert len(loaded.requests) == 1


def test_manifest_accepts_clean_request(manifest_module, tmp_path):
    path = _write_manifest(tmp_path, [{"prompt": "hello"}])
    loaded = manifest_module.load_request_manifest(path)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].prompt == "hello"


def test_manifest_rejects_error_after_valid_line(manifest_module, tmp_path):
    path = _write_manifest(
        tmp_path,
        [
            {"prompt": "hello"},
            {"prompt": "world", "error": {"message": "boom", "code": 500}},
        ],
    )
    with pytest.raises(ValueError, match="stream error"):
        manifest_module.load_request_manifest(path)


def test_manifest_error_message_is_readable(manifest_module, tmp_path):
    path = _write_manifest(
        tmp_path,
        [{"prompt": "hello", "error": {"message": "boom", "code": 500}}],
    )
    with pytest.raises(ValueError) as excinfo:
        manifest_module.load_request_manifest(path)
    assert "boom" in str(excinfo.value)


def test_manifest_rejects_error_in_sampling_params(manifest_module, tmp_path):
    path = _write_manifest(
        tmp_path,
        [
            {
                "prompt": "hello",
                "sampling_params": {"error": {"message": "boom", "code": 500}},
            }
        ],
    )
    with pytest.raises(ValueError, match="stream error"):
        manifest_module.load_request_manifest(path)
