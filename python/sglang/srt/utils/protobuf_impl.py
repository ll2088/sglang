"""Protobuf runtime guidance shared by SGLang entrypoints.

Native protobuf (``upb``/``cpp``) is significantly faster than the pure
Python implementation for gRPC/streaming payloads.  The active implementation
is fixed at import time, so the environment variable must be set before
``google.protobuf`` is imported.
"""

import logging

logger = logging.getLogger(__name__)

_WARNED = False


def get_protobuf_implementation():
    """Return the active protobuf implementation name, or ``None``.

    Detection is best-effort: if the private API is unavailable the caller
    must continue startup without a warning.
    """
    try:
        from google.protobuf.internal import api_implementation

        return api_implementation.Type()
    except (ImportError, AttributeError):
        return None


def warn_if_python_protobuf() -> None:
    """Warn once when the pure Python protobuf implementation is active.

    Never raises and never blocks startup; ``upb`` and ``cpp`` are silent.
    """
    global _WARNED
    if _WARNED:
        return
    implementation = get_protobuf_implementation()
    if implementation != "python":
        return
    _WARNED = True
    logger.warning(
        "SGLang is using the pure Python protobuf implementation, which can "
        "reduce gRPC/streaming throughput. Set "
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb before starting Python "
        "and install a protobuf wheel with the native extension. Changing "
        "the environment after importing protobuf does not change the "
        "active implementation."
    )
