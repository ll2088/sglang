"""Unit tests for the optional protobuf implementation detection."""

import builtins
import importlib
import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("implementation", ["python", "upb"])
def test_protobuf_warning_in_fresh_process(implementation: str) -> None:
    script = (
        "import os, sys\n"
        "from unittest.mock import patch\n"
        "from google.protobuf.internal import api_implementation\n"
        "from sglang.srt.utils import protobuf_impl\n"
        "assert api_implementation.Type() == sys.argv[1]\n"
        "os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "
        "'upb' if sys.argv[1] == 'python' else 'python'\n"
        "with patch.object(protobuf_impl.logger, 'warning') as warning:\n"
        "    protobuf_impl.warn_if_python_protobuf()\n"
        "    if sys.argv[1] == 'python':\n"
        "        warning.assert_called_once()\n"
        "        assert 'before starting Python' in warning.call_args.args[0]\n"
        "    else:\n"
        "        warning.assert_not_called()\n"
        "assert api_implementation.Type() == sys.argv[1]\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, implementation],
        env={**os.environ, "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": implementation},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cpp_protobuf_does_not_warn(monkeypatch) -> None:
    from google.protobuf.internal import api_implementation

    from sglang.srt.utils import protobuf_impl

    warning = MagicMock()
    monkeypatch.setattr(api_implementation, "Type", lambda: "cpp")
    monkeypatch.setattr(protobuf_impl.logger, "warning", warning)
    protobuf_impl.warn_if_python_protobuf()
    warning.assert_not_called()


@pytest.mark.parametrize("missing_api", ["import", "Type"])
def test_protobuf_warning_skips_unavailable_api(monkeypatch, missing_api: str) -> None:
    from google.protobuf.internal import api_implementation

    from sglang.srt.utils import protobuf_impl

    warning = MagicMock()
    monkeypatch.setattr(protobuf_impl.logger, "warning", warning)
    if missing_api == "import":
        real_import = builtins.__import__

        def import_without_implementation(
            name, globals=None, locals=None, fromlist=(), level=0
        ):
            if name == "google.protobuf.internal" and "api_implementation" in fromlist:
                raise ImportError("protobuf implementation API unavailable")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", import_without_implementation)
    else:
        monkeypatch.delattr(api_implementation, "Type")

    importlib.reload(protobuf_impl)
    protobuf_impl.warn_if_python_protobuf()
    warning.assert_not_called()
