"""Startup diagnostics must never block model startup."""

from unittest.mock import MagicMock

import pytest

from sglang.srt.utils import protobuf_impl


@pytest.mark.parametrize("implementation", ["python", "upb", "cpp", None])
def test_warning_never_raises(monkeypatch, implementation) -> None:
    warning = MagicMock()
    monkeypatch.setattr(protobuf_impl.logger, "warning", warning)
    monkeypatch.setattr(
        protobuf_impl, "get_protobuf_implementation", lambda: implementation
    )
    protobuf_impl.warn_if_python_protobuf()
    if implementation == "python":
        warning.assert_called_once()
    else:
        warning.assert_not_called()
