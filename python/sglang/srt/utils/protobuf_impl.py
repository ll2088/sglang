"""Protobuf runtime guidance shared by the gRPC / streaming entrypoints.

Native protobuf (``upb`` or ``cpp``) is significantly faster than the pure
Python implementation for serializing streaming responses.  This module only
*diagnoses* the active implementation: detection is best-effort and must never
prevent startup, so any failure to import or introspect the private protobuf
API is silently ignored.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_WARNED = False


def get_protobuf_implementation() -> str | None:
    """Return the active protobuf implementation name, or ``None``.

    ``None`` means the private implementation API is unavailable (older or
    stripped protobuf wheels); callers must treat that as "unknown" and skip
    any diagnostics rather than failing.
    """
    try:
        from google.protobuf.internal import api_implementation

        implementation = api_implementation.Type()
    except (ImportError, AttributeError):
        return None
    if not isinstance(implementation, str):
        return None
    return implementation


def warn_if_python_protobuf() -> None:
    """Warn once when the pure-Python protobuf implementation is active.

    Native ``upb`` and ``cpp`` implementations do not warn.  If the
    implementation cannot be detected, this is a no-op so that startup is
    never blocked by a diagnostic.
    """
    global _WARNED
    if _WARNED:
        return
    implementation = get_protobuf_implementation()
    if implementation != "python":
        return
    _WARNED = True
    logger.warning(
        "SGLang is using the pure-Python protobuf implementation, which can "
        "reduce streaming throughput. Set "
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb before starting Python "
        "and install a protobuf wheel that ships the native extension. "
        "Changing the environment after protobuf has been imported does not "
        "change the active implementation."
    )
