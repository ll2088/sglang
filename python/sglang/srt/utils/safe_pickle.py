# SPDX-License-Identifier: Apache-2.0
"""Safe pickle loading helpers.

Only plain Python containers and primitives are allowed by default.  Callers
that need to load a trusted archive must supply an explicit SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import io
import pickle
from typing import Any, Optional


class UnsafePickleError(RuntimeError):
    """Raised when a pickle payload is rejected by the safety checks."""


_ALLOWED_GLOBALS = frozenset(
    {
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
)


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_GLOBALS:
            raise UnsafePickleError(
                f"pickle payload references disallowed global {module}.{name}"
            )
        return super().find_class(module, name)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_pickle_load(path: str, expected_sha256: Optional[str] = None) -> Any:
    """Load ``path`` with a restricted unpickler.

    If ``expected_sha256`` is provided the file digest must match exactly
    (case-insensitive).  Otherwise only whitelisted globals are permitted.
    """
    with open(path, "rb") as handle:
        data = handle.read()

    if expected_sha256 is not None:
        actual = _sha256(data)
        if actual.lower() != expected_sha256.lower():
            raise UnsafePickleError(
                f"pickle digest mismatch for {path}: expected {expected_sha256}, got {actual}"
            )

    try:
        return _RestrictedUnpickler(io.BytesIO(data)).load()
    except UnsafePickleError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize to a single error type
        raise UnsafePickleError(f"failed to load pickle {path}: {exc}") from exc
