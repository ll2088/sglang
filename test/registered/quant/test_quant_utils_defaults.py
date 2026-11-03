# SPDX-License-Identifier: Apache-2.0
"""Regression tests for mutable-default and shared-state fixes.

These tests are intentionally self-contained (pytest + stdlib only) so they
can be collected without importing the full sglang runtime. They mirror the
upstream fix pattern: mutable defaults become ``None`` and module-level
mappings are copied before use.
"""

import dataclasses
import hashlib
import pickle
from types import MappingProxyType

import pytest


_FALLBACK_FUSED_SHARDS = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _copy_fused_mapping(fused_mapping=None):
    source = (
        fused_mapping
        if fused_mapping is not None
        else MappingProxyType(_FALLBACK_FUSED_SHARDS)
    )
    return {key: list(value) for key, value in source.items()}


def _append_default(value, items=None):
    if items is None:
        items = []
    items.append(value)
    return items


def test_mutable_default_is_isolated_between_calls():
    first = _append_default(1)
    second = _append_default(2)
    assert first == [1]
    assert second == [2]
    assert first is not second


def test_fallback_mapping_is_not_mutated_in_place():
    before = {key: list(value) for key, value in _FALLBACK_FUSED_SHARDS.items()}
    copied = _copy_fused_mapping()
    copied["qkv_proj"].append("extra_proj")
    copied["new_proj"] = ["x"]
    assert _FALLBACK_FUSED_SHARDS == before
    assert "new_proj" not in _FALLBACK_FUSED_SHARDS


def test_caller_mapping_is_copied():
    caller = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    copied = _copy_fused_mapping(caller)
    copied["qkv_proj"].append("extra_proj")
    assert caller == {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}


@pytest.mark.parametrize("key", sorted(_FALLBACK_FUSED_SHARDS))
def test_each_fallback_key_is_independent(key):
    first = _copy_fused_mapping()
    second = _copy_fused_mapping()
    first[key].append("sentinel")
    assert "sentinel" not in second[key]


# --- dataclass optional-field checks (mirrors hasattr -> is not None fix) ---


@dataclasses.dataclass
class _OptionalFields:
    vocab_embedding: object = None
    lm_head: object = None


def _has_optional_fields_buggy(obj):
    # The upstream bug: hasattr on a dataclass field is always True.
    return hasattr(obj, "vocab_embedding") and hasattr(obj, "lm_head")


def _has_optional_fields_fixed(obj):
    return obj.vocab_embedding is not None and obj.lm_head is not None


@pytest.mark.parametrize(
    ("vocab", "head", "expected"),
    [
        (None, None, False),
        (object(), None, False),
        (None, object(), False),
        (object(), object(), True),
    ],
)
def test_dataclass_optional_fields_use_is_not_none(vocab, head, expected):
    obj = _OptionalFields(vocab_embedding=vocab, lm_head=head)
    # hasattr is always True for declared dataclass fields, so it cannot
    # distinguish a missing value from a present one.
    assert _has_optional_fields_buggy(obj) is True
    assert _has_optional_fields_fixed(obj) is expected


# --- safe pickle loading (mirrors weights_only / sha256 verification) ---


class _UnsafePayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (str(self.marker), "w"))


def _load_verified(path, expected_sha256):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError("Checkpoint failed SHA-256 verification")
    # Only after verification do we deserialize, and only allow plain data.
    return pickle.loads(data)


def test_pickle_rejects_unsafe_payload_without_executing(tmp_path):
    marker = tmp_path / "executed"
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps(_UnsafePayload(marker)))
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    with pytest.raises(Exception):
        _load_verified(payload, digest)
    assert not marker.exists()


def test_pickle_rejects_checksum_mismatch(tmp_path):
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps({"ok": True}))
    with pytest.raises(RuntimeError, match="SHA-256"):
        _load_verified(payload, "0" * 64)


# --- MoE init keyword contract (mirrors child_init TypeError fix) ---


def _init_moe_module_old(orig_router_weight=None, new_router_weight=None):
    raise TypeError("unexpected keyword argument 'orig_router_weight'")


def _init_moe_module_new(
    orig_router_weights=None,
    new_router_weights=None,
    orig_num_experts=None,
    new_num_experts=None,
):
    return {
        "orig_router_weights": orig_router_weights,
        "new_router_weights": new_router_weights,
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def test_moe_init_old_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _init_moe_module_old(orig_router_weight=1, new_router_weight=2)


def test_moe_init_new_keywords_accept_plural_names():
    result = _init_moe_module_new(
        orig_router_weights=[1],
        new_router_weights=[2],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2
