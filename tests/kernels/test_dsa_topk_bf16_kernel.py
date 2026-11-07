"""Standalone tests for the indexer_topk backend-resolution helpers.

These tests avoid importing torch/CUDA by exercising the pure-Python helpers
and by monkeypatching the runtime context lookups.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from sglang.srt.state_capturer import indexer_topk as it


def test_extract_indexer_topk_from_meta_info_roundtrip():
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    encoded = base64.b64encode(values.tobytes()).decode("utf-8")
    data = {"meta_info": {"indexer_topk": encoded}}
    out = it.extract_indexer_topk_from_meta_info(data)
    np.testing.assert_array_equal(out, values)


def test_extract_indexer_topk_missing_returns_empty():
    out = it.extract_indexer_topk_from_meta_info({"meta_info": {}})
    assert out.dtype == np.int32
    assert out.size == 0


def test_maybe_capture_passthrough_without_capturer(monkeypatch):
    monkeypatch.setattr(it, "get_global_indexer_capturer", lambda: None)
    sentinel = object()
    assert it.maybe_capture_indexer_topk(0, sentinel) is sentinel
    assert it.maybe_capture_indexer_topk(0, None) is None


def test_maybe_capture_calls_capturer(monkeypatch):
    calls = []

    class _Cap:
        def capture(self, layer_id, topk_indices):
            calls.append((layer_id, topk_indices))

    monkeypatch.setattr(it, "get_global_indexer_capturer", lambda: _Cap())
    sentinel = object()
    assert it.maybe_capture_indexer_topk(3, sentinel) is sentinel
    assert calls == [(3, sentinel)]


def test_create_indexer_capturer_disabled_returns_none(monkeypatch):
    class _Exec:
        class features:
            enable_return_indexer_topk = False

    monkeypatch.setattr(it, "get_exec", lambda: _Exec())
    out = it._create_indexer_capturer_raw(
        enable=False,
        num_indexer_layers=4,
        index_topk=2048,
        num_tokens=16,
        max_running_requests=8,
        device="cuda",
    )
    assert out is None


def test_create_indexer_capturer_invalid_topk_returns_none(monkeypatch):
    out = it._create_indexer_capturer_raw(
        enable=True,
        num_indexer_layers=4,
        index_topk=0,
        num_tokens=16,
        max_running_requests=8,
        device="cuda",
    )
    assert out is None


def test_create_indexer_capturer_no_layers_returns_none():
    out = it._create_indexer_capturer_raw(
        enable=True,
        num_indexer_layers=0,
        index_topk=2048,
        num_tokens=16,
        max_running_requests=8,
        device="cuda",
    )
    assert out is None
