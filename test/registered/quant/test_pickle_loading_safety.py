# SPDX-License-Identifier: Apache-2.0
"""Regression tests for restricted pickle loading.

These tests exercise a small, dependency-free restricted unpickler that mirrors
the safety contract used by model-weight loaders: only plain containers and
primitive scalars are allowed, and any attempt to resolve a callable must be
rejected *before* it is invoked.
"""

import io
import os
import pickle

import pytest


class _RestrictedUnpickler(pickle.Unpickler):
    """Allow only data-only opcodes; refuse every global/class lookup."""

    _ALLOWED = {
        ("builtins", "dict"),
        ("builtins", "list"),
        ("builtins", "tuple"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "str"),
        ("builtins", "bytes"),
        ("builtins", "bool"),
        ("builtins", "complex"),
        ("builtins", "bytearray"),
        ("collections", "OrderedDict"),
    }

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to load global {module}.{name} from untrusted pickle"
        )


def _restricted_loads(payload: bytes):
    return _RestrictedUnpickler(io.BytesIO(payload)).load()


class _MaliciousPayload:
    """A payload whose unpickling would create a marker file if executed."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def test_restricted_unpickler_round_trips_plain_data():
    data = {"a": [1, 2, 3], "b": ("x", b"y"), "c": {1.5, 2.5}}
    assert _restricted_loads(pickle.dumps(data)) == data


def test_restricted_unpickler_rejects_malicious_payload(tmp_path):
    marker = tmp_path / "executed"
    payload = pickle.dumps(_MaliciousPayload(marker))
    with pytest.raises(pickle.UnpicklingError):
        _restricted_loads(payload)
    assert not marker.exists()


def test_restricted_unpickler_rejects_unknown_global():
    payload = pickle.dumps(os.getcwd)
    with pytest.raises(pickle.UnpicklingError):
        _restricted_loads(payload)


def test_restricted_unpickler_rejects_truncated_payload():
    with pytest.raises((pickle.UnpicklingError, EOFError)):
        _restricted_loads(b"\x80\x04\x95")
