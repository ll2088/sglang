"""Tests for the optional protobuf implementation warning."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sglang.srt.utils import protobuf_impl


def test_python_protobuf_warns(monkeypatch) -> None:
    warning = MagicMock()
    monkeypatch.setattr(protobuf_impl, "get_protobuf_implementation", lambda: "python")
    monkeypatch.setattr(protobuf_impl.logger, "warning", warning)
    protobuf_impl.warn_if_python_protobuf()
    warning.assert_called_once()
    message = warning.call_args.args[0]
    assert "before starting Python" in message
    assert "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb" in message


@pytest.mark.parametrize("implementation", ["upb", "cpp", None])
def test_native_or_unknown_protobuf_does_not_warn(monkeypatch, implementation) -> None:
    warning = MagicMock()
    monkeypatch.setattr(
        protobuf_impl, "get_protobuf_implementation", lambda: implementation
    )
    monkeypatch.setattr(protobuf_impl.logger, "warning", warning)
    protobuf_impl.warn_if_python_protobuf()
    warning.assert_not_called()


def test_get_implementation_returns_none_when_api_missing(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def import_without_implementation(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google.protobuf.internal" and "api_implementation" in fromlist:
            raise ImportError("protobuf implementation API unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_implementation)
    assert protobuf_impl.get_protobuf_implementation() is None
