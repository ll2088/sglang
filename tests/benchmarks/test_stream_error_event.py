"""Tests for stream error event handling in benchmark clients.

The benchmark helpers are loaded by file path so this test does not import the
``sglang`` package (and its heavy runtime dependencies) just to exercise pure
parsing helpers.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCH_DIR = _REPO_ROOT / "python" / "sglang" / "multimodal_gen" / "benchmarks"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_request_manifest = _load_module(
    "_sglang_request_manifest_for_tests", _BENCH_DIR / "request_manifest.py"
)
load_request_manifest = _request_manifest.load_request_manifest


def _extract_stream_error(payload):
    """Mirror of bench_serving._extract_stream_error (kept dependency-free)."""
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if error is None:
        return None
    if isinstance(error, str):
        return error
    return json.dumps(error)


def _get_response_output_count(resp_json):
    """Mirror of bench_serving._get_response_output_count."""
    if isinstance(resp_json.get("num_outputs"), int):
        return resp_json["num_outputs"]
    if isinstance(resp_json.get("data"), list):
        return len(resp_json["data"])
    if isinstance(resp_json.get("file_paths"), list):
        return len(resp_json["file_paths"])
    if isinstance(resp_json.get("urls"), list):
        return len(resp_json["urls"])
    if resp_json.get("file_path") or resp_json.get("url"):
        return 1
    return 0


STREAM_ERROR_MESSAGE = "internal server error during generation"


def _error_payload(message: str = STREAM_ERROR_MESSAGE) -> dict:
    return {
        "error": {
            "message": message,
            "type": "InternalServerError",
            "param": None,
            "code": 500,
        }
    }


def test_extract_stream_error_returns_server_error() -> None:
    payload = _error_payload()
    error = _extract_stream_error(payload)
    assert error is not None
    assert STREAM_ERROR_MESSAGE in error


def test_extract_stream_error_ignores_normal_chunks() -> None:
    assert _extract_stream_error({"choices": [{"text": "a"}]}) is None
    assert _extract_stream_error({"choices": [{"text": "b"}]}) is None
    assert _extract_stream_error({}) is None
    assert _extract_stream_error(None) is None


def test_extract_stream_error_accepts_string_error() -> None:
    assert _extract_stream_error({"error": "boom"}) == "boom"


def test_request_manifest_rejects_stream_error(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "hello", "error": _error_payload()["error"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stream error"):
        load_request_manifest(manifest)


def test_request_manifest_accepts_normal_request(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "hello", "sampling_params": {"width": 64}}) + "\n",
        encoding="utf-8",
    )
    loaded = load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].prompt == "hello"


def test_extract_stream_error_ignores_usage_only_chunk() -> None:
    # A trailing usage summary carries no error and must not fail the request.
    assert _extract_stream_error({"usage": {"completion_tokens": 2}}) is None


def test_response_output_count_ignores_error_payload() -> None:
    # An error envelope carries no outputs, so it must not be counted.
    assert _get_response_output_count(_error_payload()) == 0


def _classify_stream(events: list[str]) -> tuple[bool, str | None]:
    """Mirror of the benchmark client rule for a streamed SSE event list.

    Returns ``(success, error)``. A stream that delivered tokens and then an
    error envelope must be reported as failed, with the server error kept.
    """
    saw_token = False
    stream_error: str | None = None
    for event in events:
        if event == "[DONE]":
            continue
        payload = json.loads(event)
        error = _extract_stream_error(payload)
        if error is not None:
            stream_error = error
        elif payload.get("choices"):
            saw_token = True
    if stream_error is not None:
        return False, stream_error
    return saw_token, None


def test_stream_with_tokens_then_error_is_failed() -> None:
    events = [
        json.dumps({"choices": [{"text": "a"}]}),
        json.dumps({"choices": [{"text": "b"}]}),
        json.dumps(_error_payload()),
        "[DONE]",
    ]
    success, error = _classify_stream(events)
    assert not success
    assert error is not None
    assert STREAM_ERROR_MESSAGE in error


def test_stream_without_error_is_successful() -> None:
    events = [
        json.dumps({"choices": [{"text": "a"}]}),
        json.dumps({"choices": [{"text": "b"}]}),
        json.dumps({"usage": {"completion_tokens": 2}}),
        "[DONE]",
    ]
    success, error = _classify_stream(events)
    assert success
    assert error is None


def test_stream_without_tokens_is_failed() -> None:
    events = [json.dumps({"usage": {"completion_tokens": 0}}), "[DONE]"]
    success, error = _classify_stream(events)
    assert not success
    assert error is None


def test_offline_generation_error_is_failed() -> None:
    """bench_offline_throughput treats a result carrying ``error`` as failed."""

    class _Result:
        def __init__(self, error):
            self.error = error
            self.output_file_path = None

    results = [_Result(None), _Result(STREAM_ERROR_MESSAGE)]
    stream_error = next(
        (getattr(item, "error", None) for item in results if getattr(item, "error", None)),
        None,
    )
    assert stream_error is not None
    assert STREAM_ERROR_MESSAGE in stream_error


def test_summarize_skips_error_results() -> None:
    """A benchmark result carrying an error must not be counted as success."""
    data = {"error": _error_payload()["error"], "aggregated_metrics": {"stats": {}}}
    assert data.get("error") is not None
