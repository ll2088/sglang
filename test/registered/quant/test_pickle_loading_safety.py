# SPDX-License-Identifier: Apache-2.0
"""Regression tests for safe pickle loading.

These tests are intentionally self-contained: they do not import ``sglang`` so
that they can run without the full runtime dependency tree. The helper below
mirrors the safe-loading contract used by model weight loaders: only tensors
and plain containers are accepted, and any other pickle object is rejected
before it can execute code.
"""

import io
import os
import pickle

import pytest
import torch


_ALLOWED_PICKLE_GLOBALS = {
    ("collections", "OrderedDict"),
    ("torch", "Tensor"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch.storage", "_load_from_bytes"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED_PICKLE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to load pickle global {module}.{name}"
        )


def safe_load_pickle(data: bytes):
    """Load a pickle payload while rejecting non-whitelisted globals."""
    return _RestrictedUnpickler(io.BytesIO(data)).load()


class _MaliciousPayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.mkdir, (str(self.marker),)


def test_safe_load_accepts_tensor_payload():
    tensor = torch.arange(4)
    payload = pickle.dumps(tensor)
    restored = safe_load_pickle(payload)
    assert torch.equal(restored, tensor)


def test_safe_load_rejects_malicious_payload_without_executing(tmp_path):
    marker = tmp_path / "executed"
    payload = pickle.dumps(_MaliciousPayload(marker))
    with pytest.raises(pickle.UnpicklingError):
        safe_load_pickle(payload)
    assert not marker.exists()


def test_safe_load_rejects_unknown_global():
    payload = pickle.dumps({"fn": os.system})
    with pytest.raises(pickle.UnpicklingError):
        safe_load_pickle(payload)
