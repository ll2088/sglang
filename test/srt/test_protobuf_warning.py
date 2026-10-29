import builtins
import importlib

import pytest

from sglang.srt.utils import protobuf_impl


@pytest.mark.parametrize("implementation", ["python", "upb", "cpp"])
def test_warn_only_for_python(monkeypatch, implementation):
    from google.protobuf.internal import api_implementation

    warnings = []
    monkeypatch.setattr(api_implementation, "Type", lambda: implementation)
    monkeypatch.setattr(
        protobuf_impl.logger, "warning", lambda msg, *a, **k: warnings.append(msg)
    )
    protobuf_impl.warn_if_python_protobuf()
    if implementation == "python":
        assert len(warnings) == 1
        assert "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb" in warnings[0]
    else:
        assert warnings == []


@pytest.mark.parametrize("missing", ["import", "Type"])
def test_warn_skips_unavailable_api(monkeypatch, missing):
    from google.protobuf.internal import api_implementation

    warnings = []
    monkeypatch.setattr(
        protobuf_impl.logger, "warning", lambda msg, *a, **k: warnings.append(msg)
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
    assert warnings == []


def test_warn_is_idempotent(monkeypatch):
    from google.protobuf.internal import api_implementation

    warnings = []
    monkeypatch.setattr(api_implementation, "Type", lambda: "python")
    monkeypatch.setattr(
        protobuf_impl.logger, "warning", lambda msg, *a, **k: warnings.append(msg)
    )
    protobuf_impl.warn_if_python_protobuf()
    protobuf_impl.warn_if_python_protobuf()
    assert len(warnings) == 1
