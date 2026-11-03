# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

Unsafe pickle payloads must be rejected without executing code. The helper
below mirrors the hardened loading pattern: only allow-listed globals are
permitted, and a SHA-256 digest is required for externally supplied weights.

Self-contained (stdlib + pytest only); does not import sglang.
"""

from __future__ import annotations

import hashlib
import io
import pickle

import pytest


_ALLOWED_GLOBALS = {
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "int"),
    ("builtins", "str"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"Refusing to load global {module}.{name} from untrusted pickle"
            )
        return super().find_class(module, name)


def _safe_loads(payload: bytes):
    return _RestrictedUnpickler(io.BytesIO(payload)).load()


def _verify_sha256(payload: bytes, expected_hex: str) -> None:
    actual = hashlib.sha256(payload).hexdigest()
    if actual.lower() != expected_hex.lower():
        raise RuntimeError("Supplied weights failed SHA-256 verification")


class _Malicious:
    def __reduce__(self):
        return (eval, ("__import__('os').getcwd()",))


def test_malicious_payload_is_rejected_without_execution():
    payload = pickle.dumps(_Malicious())
    with pytest.raises(pickle.UnpicklingError):
        _safe_loads(payload)


def test_allow_listed_payload_round_trips():
    payload = pickle.dumps({"a": [1, 2, 3]})
    assert _safe_loads(payload) == {"a": [1, 2, 3]}


def test_checksum_mismatch_is_rejected():
    payload = b"trusted checkpoint"
    with pytest.raises(RuntimeError, match="SHA-256"):
        _verify_sha256(payload, "0" * 64)


def test_checksum_match_is_accepted():
    payload = b"trusted checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    _verify_sha256(payload, digest.upper())
