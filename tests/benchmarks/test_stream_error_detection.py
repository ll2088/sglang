"""Centralized tests for stream error-event handling in benchmark clients.

These tests avoid importing ``sglang`` so they can run in a lightweight
environment without the full serving dependency tree. They mirror the upstream
rule: a stream that delivered tokens and then an ``error`` envelope must be
reported as failed, while a clean stream stays successful.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

try:
    import pytest
except ImportError:  # pragma: no cover - lightweight CI without pytest
    pytest = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARKS_DIR = (
    _REPO_ROOT / "python" / "sglang" / "multimodal_gen" / "benchmarks"
)
_MANIFEST_PATH = _BENCHMARKS_DIR / "request_manifest.py"
_RESULTS_PATH = (
    _REPO_ROOT / "sgl-model-gateway" / "e2e_test" / "benchmarks" / "results.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_pytest():
    if pytest is None:
        raise RuntimeError("pytest is required to run these tests")
    return pytest


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


def test_error_event_stream_is_not_reported_as_success() -> None:
    _require_pytest()
    for endpoint in ("completions", "chat", "audio"):
        chunks = [
            _token_chunk("a"),
            _token_chunk("b"),
            _error_chunk(),
            b"data: [DONE]\n\n",
        ]
        success, stream_error = _classify_stream(chunks)
        assert not success, endpoint
        assert stream_error is not None, endpoint
        assert "internal server error during generation" in stream_error, endpoint


def test_clean_stream_is_reported_as_success() -> None:
    _require_pytest()
    for endpoint in ("completions", "chat", "audio"):
        chunks = [_token_chunk("a"), _token_chunk("b"), b"data: [DONE]\n\n"]
        success, stream_error = _classify_stream(chunks)
        assert success, endpoint
        assert stream_error is None, endpoint


def test_error_event_without_tokens_is_failed() -> None:
    _require_pytest()
    chunks = [_error_chunk(), b"data: [DONE]\n\n"]
    success, stream_error = _classify_stream(chunks)
    assert not success
    assert stream_error is not None


def test_manifest_rejects_stream_error(tmp_path: Path) -> None:
    _require_pytest()
    module = _load_module("_bench_request_manifest", _MANIFEST_PATH)
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
    _require_pytest()
    module = _load_module("_bench_request_manifest", _MANIFEST_PATH)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"prompt": "hello", "request_id": "r1"}) + "\n",
        encoding="utf-8",
    )
    loaded = module.load_request_manifest(manifest)
    assert len(loaded.requests) == 1
    assert loaded.requests[0].request_id == "r1"


def test_benchmark_result_rejects_stream_error(tmp_path: Path) -> None:
    _require_pytest()
    module = _load_module("_bench_results", _RESULTS_PATH)
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "error": {"message": "boom", "code": 500},
                "aggregated_metrics": {"stats": {}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stream error"):
        module.BenchmarkResult.from_json(result_path)


def test_benchmark_result_accepts_clean_run(tmp_path: Path) -> None:
    _require_pytest()
    module = _load_module("_bench_results", _RESULTS_PATH)
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "aggregated_metrics": {
                    "stats": {
                        "ttft": {"mean": 0.1},
                        "e2e_latency": {"mean": 1.0},
                        "input_throughput": {"mean": 100.0},
                        "output_throughput": {"mean": 50.0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    result = module.BenchmarkResult.from_json(result_path)
    assert result.ttft_mean == pytest.approx(0.1)
    assert result.output_throughput_mean == pytest.approx(50.0)


def test_summary_excludes_error_event_from_success(tmp_path: Path) -> None:
    """A run whose stream ended in an error event must not count as success.

    The summary is the user-facing aggregate, so a failed stream has to show
    up as a failed row rather than contributing its partial metrics.
    """
    _require_pytest()
    summarize = _load_module(
        "_bench_summarize",
        _REPO_ROOT / "sgl-model-gateway" / "e2e_test" / "benchmarks" / "summarize.py",
    )
    folder = tmp_path / "benchmark_error_case"
    folder.mkdir()
    (folder / "result.json").write_text(
        json.dumps(
            {
                "error": {"message": "boom", "code": 500},
                "aggregated_metrics": {
                    "stats": {
                        "ttft": {"mean": 0.1},
                        "e2e_latency": {"mean": 1.0},
                        "input_throughput": {"mean": 100.0},
                        "output_throughput": {"mean": 50.0},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    summary = summarize.generate_summary(tmp_path)
    assert "error case" in summary
    assert "Failed" in summary
    assert "Success" not in summary


def test_summary_counts_clean_run_as_success(tmp_path: Path) -> None:
    _require_pytest()
    summarize = _load_module(
        "_bench_summarize",
        _REPO_ROOT / "sgl-model-gateway" / "e2e_test" / "benchmarks" / "summarize.py",
    )
    folder = tmp_path / "benchmark_clean_case"
    folder.mkdir()
    (folder / "result.json").write_text(
        json.dumps(
            {
                "aggregated_metrics": {
                    "stats": {
                        "ttft": {"mean": 0.1},
                        "e2e_latency": {"mean": 1.0},
                        "input_throughput": {"mean": 100.0},
                        "output_throughput": {"mean": 50.0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    summary = summarize.generate_summary(tmp_path)
    assert "clean case" in summary
    assert "Success" in summary
    assert "Failed" not in summary
