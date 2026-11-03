# SPDX-License-Identifier: Apache-2.0
"""Small, dependency-free helpers for loading pickled data safely.

This module intentionally imports only the standard library so that it can be
used (and tested) without pulling in sglang's heavy runtime dependencies.
"""

from __future__ import annotations

import hashlib
import io
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional, Union


class UnsafePickleError(ValueError):
    """Raised when a pickle payload is rejected by the safety checks."""


# Globals that are allowed to be resolved while unpickling. Anything outside
# this set is rejected before the payload is executed.
_DEFAULT_ALLOWED_GLOBALS = frozenset(
    {
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "list"),
        ("builtins", "dict"),
        ("builtins", "tuple"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "str"),
        ("builtins", "bytes"),
        ("builtins", "bool"),
        ("builtins", "NoneType"),
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
    }
)


def sha256_file(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    """Return the lowercase hex SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(
    path: Union[str, Path],
    expected_sha256: Optional[str],
    *,
    label: str = "weights",
) -> None:
    """Verify ``path`` against ``expected_sha256``.

    ``expected_sha256`` is required: callers must supply a trusted digest for
    any file they intend to load. The comparison is case-insensitive.
    """
    if not expected_sha256:
        raise UnsafePickleError(
            f"Refusing to load {label} from {path!s}: an explicit SHA-256 "
            "digest is required."
        )
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.strip().lower():
        raise UnsafePickleError(
            f"{label.capitalize()} at {path!s} failed SHA-256 verification "
            f"(expected {expected_sha256}, got {actual})."
        )


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, file_obj: io.BufferedIOBase, allowed_globals: Iterable):
        super().__init__(file_obj)
        self._allowed = frozenset(allowed_globals)

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._allowed:
            raise UnsafePickleError(
                f"Refusing to unpickle global {module}.{name}: not in the "
                "allow-list."
            )
        return super().find_class(module, name)


def safe_pickle_loads(
    data: bytes,
    *,
    allowed_globals: Optional[Iterable] = None,
) -> Any:
    """Unpickle ``data`` while rejecting globals outside the allow-list.

    The allow-list is checked *before* the referenced callable is resolved, so
    a malicious payload cannot execute code during unpickling.
    """
    allowed = _DEFAULT_ALLOWED_GLOBALS if allowed_globals is None else allowed_globals
    return _RestrictedUnpickler(io.BytesIO(data), allowed).load()


def safe_pickle_load(
    path: Union[str, Path],
    *,
    expected_sha256: Optional[str] = None,
    *,
    allowed_globals: Optional[Iterable] = None,
    label: str = "pickle",
) -> Any:
    """Load a pickle file after verifying its SHA-256 digest.

    ``expected_sha256`` is mandatory; callers must pass a trusted digest for
    the file they are loading. The payload is then unpickled with a restricted
    unpickler that only resolves allow-listed globals.
    """
    verify_sha256(path, expected_sha256, label=label)
    with open(path, "rb") as fh:
        data = fh.read()
    return safe_pickle_loads(data, allowed_globals=allowed_globals)
