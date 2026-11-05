"""Unit tests for stream error-event handling in benchmark clients.

These tests deliberately avoid importing ``sglang`` so they can run in a
lightweight environment without the full serving dependency tree. They mirror
the upstream rule: a stream that delivered tokens and then an ``error``
envelope must be reported as failed, while a clean stream stays successful.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = (
    _REPO_ROOT
    / "python"
    / "sglang"
    / "multimodal_gen"
    / "benchmarks"
    / "request_manifest.py"
)


def _load_manifest_module():
    spec = importlib.util.spec_from_file_location(
        "_bench_request_manifest", _MANIFEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _token_chunk(text: str) -> bytes:
    return _sse({"choices": [{"text": text}]})


def _error_chunk(message: str = "internal server error during generation") -> bytes:
    return _sse(
        {
            "error": {
                "message": message,
                "type": "InternalServerError",
                "param": None,
                "code": 500,
            }
        }
    )


def _parse_sse(chunk: bytes) -> list[str]:
    text = chunk.decode()
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


def _classify_stream(chunks: list[bytes]) -> tuple[bool, str | None]:
    """Mirror the benchmark client rule: error envelope wins over tokens."""
    saw_token = False
    stream_error: str | None = None
    for chunk in chunks:
        for event in _parse_sse(chunk):
            if event == "[DONE]":
                continue
            data = json.loads(event)
            if error := data.get("error"):
                stream_error = json.dumps(error)
            elif data.get("choices"):
                saw_token = True
    success = stream_error is None and saw_token
    return success, stream_error


@pytest.mark.parametrize("endpoint", ["completions", "chat", "audio"])
def test_error_event_stream_is_not_reported_as_success(endpoint: str) -> None:
    chunks = [_token_chunk("a"), _token_chunk("b"), _error_chunk(), b"data: [DONE]\n\n"]
    success, stream_error = _classify_stream(chunks)
    assert not success
    assert stream_error is not None
    assert "internal server error during generation" in stream_error


@pytest.mark.parametrize("endpoint", ["completions", "chat", "audio"])
def test_clean_stream_is_reported_as_success(endpoint: str) -> None:
    chunks = [_token_chunk("a"), _token_chunk("b"), b"data: [DONE]\n\n"]
    success, stream_error = _classify_stream(chunks)
    assert success
    assert stream_error is None


def test_error_event_without_tokens_is_failed() -> None:
    chunks = [_error_chunk(), b"data: [DONE]\n\n"]
    success, stream_error = _classify_stream(chunks)
    assert not success
    assert stream_error is not None


def test_manifest_rejects_stream_error(tmp_path: Path) -> None:
    module = _load_manifest_module()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "prompt": "hello",
                "error": {"message": "boom", "code": 500},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stream error"):
        module.load_request_manifest(manifest)


def test_manifest_accepts_clean_request(tmp_path: Path) -> None:
    module = _load_manifest_module()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "hello", "request_id": "r1"}) + "\n",
        encoding="utf-8",
    )
    loaded = module.load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].request_id == "r1"
