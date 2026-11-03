# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe checkpoint loading.

A malicious pickle payload must be rejected without executing code. The
loader under test is a small, dependency-free helper that mirrors the
`weights_only=True` policy used by the model loader.
"""

import io
import os
import pickle

import pytest


class _RestrictedUnpickler(pickle.Unpickler):
    """Allow only plain containers/scalars; reject anything that can execute code."""

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
        ("builtins", "NoneType"),
    }

    def find_class(self, module, name):
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(
                f"Refusing to load {module}.{name} from checkpoint"
            )
        return super().find_class(module, name)


class _MaliciousPayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def _safe_load(buffer: io.BytesIO):
    """Load a checkpoint with the restricted unpickler policy."""
    return _RestrictedUnpickler(buffer, fix_imports=False).load()


def test_malicious_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    payload = pickle.dumps(_MaliciousPayload(marker))
    with pytest.raises(Exception):
        _safe_load(io.BytesIO(payload))
    assert not marker.exists()


def test_plain_dict_pickle_roundtrips():
    data = {"weight": [1, 2, 3]}
    assert _safe_load(io.BytesIO(pickle.dumps(data))) == data
