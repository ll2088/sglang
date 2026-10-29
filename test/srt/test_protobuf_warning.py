import builtins
import importlib

import pytest

from sglang.srt.utils import protobuf_impl


@pytest.mark.parametrize("implementation", ["python", "upb", "cpp"])
def test_warn_only_for_python(monkeypatch, implementation):
    from google.protobuf.internal import api_implementation

    calls = []
    monkeypatch.setattr(api_implementation, "Type", lambda: implementation)
    monkeypatch.setattr(
        protobuf_impl.logger, "warning", lambda *a, **k: calls.append(a)
    )
    protobuf_impl.warn_if_python_protobuf()
    if implementation == "python":
        assert len(calls) == 1
        assert "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb" in calls[0][0]
    else:
        assert calls == []


@pytest.mark.parametrize("missing", ["import", "Type"])
def test_missing_api_does_not_raise(monkeypatch, missing):
    from google.protobuf.internal import api_implementation

    calls = []
    monkeypatch.setattr(
        protobuf_impl.logger, "warning", lambda *a, **k: calls.append(a)
    )
    if missing == "import":
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "google.protobuf.internal" and "api_implementation" in fromlist:
                raise ImportError("unavailable")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
    else:
        monkeypatch.delattr(api_implementation, "Type")
    importlib.reload(protobuf_impl)
    protobuf_impl.warn_if_python_protobuf()
    assert calls == []
