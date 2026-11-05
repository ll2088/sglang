"""Manifest entries carrying a stream error must be rejected."""

import json

import pytest

from sglang.multimodal_gen.benchmarks.request_manifest import load_request_manifest


def _write_manifest(tmp_path, records):
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_manifest_rejects_top_level_error(tmp_path):
    path = _write_manifest(
        tmp_path,
        [{"prompt": "a cat", "error": {"message": "boom", "code": 500}}],
    )
    with pytest.raises(ValueError, match="stream error"):
        load_request_manifest(path)


def test_manifest_rejects_sampling_params_error(tmp_path):
    path = _write_manifest(
        tmp_path,
        [{"prompt": "a cat", "sampling_params": {"error": "boom"}}],
    )
    with pytest.raises(ValueError, match="stream error"):
        load_request_manifest(path)


def test_manifest_accepts_clean_record(tmp_path):
    path = _write_manifest(
        tmp_path,
        [{"prompt": "a cat", "sampling_params": {"width": 64, "height": 64}}],
    )
    manifest = load_request_manifest(path)
    assert len(manifest.requests) == 1
    assert manifest.requests[0].prompt == "a cat"
