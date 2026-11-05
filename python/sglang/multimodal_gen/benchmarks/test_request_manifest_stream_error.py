"""Tests that request manifests reject records carrying a stream error."""

import json

import pytest

from sglang.multimodal_gen.benchmarks.request_manifest import load_request_manifest


def _write_manifest(tmp_path, records):
    path = tmp_path / "workload.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_manifest_with_error_object_is_rejected(tmp_path):
    path = _write_manifest(
        tmp_path,
        [
            {"prompt": "ok"},
            {
                "prompt": "failed",
                "error": {"message": "boom", "type": "InternalServerError", "code": 500},
            },
        ],
    )

    with pytest.raises(ValueError, match="stream error"):
        load_request_manifest(path)


def test_manifest_with_error_string_is_rejected(tmp_path):
    path = _write_manifest(tmp_path, [{"prompt": "failed", "error": "boom"}])

    with pytest.raises(ValueError, match="boom"):
        load_request_manifest(path)


def test_manifest_without_error_is_accepted(tmp_path):
    path = _write_manifest(tmp_path, [{"prompt": "ok"}, {"prompt": "also ok"}])

    manifest = load_request_manifest(path)

    assert len(manifest.requests) == 2
