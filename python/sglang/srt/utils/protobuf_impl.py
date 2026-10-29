"""Protobuf runtime guidance shared by the gRPC / SMG launchers.

This module only uses the standard library plus an optional import of the
protobuf implementation API. Detection is best-effort: when the private API
cannot be imported or has no ``Type`` attribute, callers skip the warning and
continue startup.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Exposed for tests that need to swap the logger without importing the
# ``sglang`` package ``__init__`` (which pulls in heavy runtime deps).
__all__ = [
    "get_protobuf_implementation",
    "warn_if_python_protobuf",
    "logger",
]

_WARNING_MESSAGE = (
    "SGLang gRPC/SMG is using Python protobuf, which can reduce throughput. "
    "Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb before starting Python "
    "and install a supported protobuf wheel with the upb extension. "
    "Changing the environment after importing protobuf does not change "
    "the active implementation."
)


def get_protobuf_implementation() -> str | None:
    """Return the active protobuf implementation name, or ``None``.

    ``None`` means the private implementation API is unavailable; callers must
    treat that as "unknown" and skip any diagnostic rather than failing.
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
    """Warn once for pure Python protobuf when it can be detected.

    Native ``upb`` and ``cpp`` implementations do not warn. An unavailable
    detection API never blocks startup.
    """
    implementation = get_protobuf_implementation()
    if implementation == "python":
        logger.warning(_WARNING_MESSAGE)
