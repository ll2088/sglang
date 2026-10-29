"""Protobuf implementation warning tests (standard library only)."""

import importlib.util
import pathlib
import sys
import types

import pytest


def _load_protobuf_impl():
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "python"
        / "sglang"
        / "srt"
        / "utils"
        / "protobuf_impl.py"
    )
    spec = importlib.util.spec_from_file_location("sglang_protobuf_impl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
 def protobuf_impl():
    return _load_protobuf_impl()


def _install_fake_api(monkeypatch, implementation):
    api = types.SimpleNamespace(Type=lambda: implementation)
    internal = types.ModuleType("google.protobuf.internal")
    internal.api_implementation = api
    google = types.ModuleType("google")
    protobuf = types.ModuleType("google.protobuf")
    protobuf.internal = internal
    google.protobuf = protobuf
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.protobuf", protobuf)
    monkeypatch.setitem(sys.modules, "google.protobuf.internal", internal)


def test_python_implementation_warns(protobuf_impl, monkeypatch):
    _install_fake_api(monkeypatch, "python")
    warnings = []
    monkeypatch.setattr(protobuf_impl.logger, "warning", warnings.append)
    protobuf_impl.warn_if_python_protobuf()
    assert len(warnings) == 1
    assert "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb" in warnings[0]


def test_native_implementations_do_not_warn(protobuf_impl, monkeypatch):
    for implementation in ("upb", "cpp"):
        _install_fake_api(monkeypatch, implementation)
        warnings = []
        monkeypatch.setattr(protobuf_impl.logger, "warning", warnings.append)
        protobuf_impl.warn_if_python_protobuf()
        assert warnings == []


def test_missing_api_does_not_raise(protobuf_impl, monkeypatch):
    monkeypatch.setitem(sys.modules, "google", None)
    monkeypatch.setitem(sys.modules, "google.protobuf", None)
    monkeypatch.setitem(sys.modules, "google.protobuf.internal", None)
    protobuf_impl.warn_if_python_protobuf()
