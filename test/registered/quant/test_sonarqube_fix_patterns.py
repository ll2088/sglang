# SPDX-License-Identifier: Apache-2.0
"""Standalone regression tests for the SonarQube fix patterns ported from
NVIDIA/Model-Optimizer PR #3063.

These tests intentionally avoid importing heavy sglang runtime modules so they
can run in a minimal CI environment.  They exercise the *patterns* that the
upstream PR fixed:

1. mutable default arguments (dict/list/set) shared across calls
2. module-level dict mutated in place
3. dataclass optional field checked with hasattr instead of `is not None`
4. unsafe pickle loading without checksum verification
5. MoE init keyword mismatch raising TypeError
"""

import hashlib
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# 1. Mutable default arguments
# ---------------------------------------------------------------------------


def _good_merge(base: Optional[Dict[str, Any]] = None, extra: Optional[List[int]] = None):
    """Fixed version: defaults are None and a fresh container is created."""
    base = {} if base is None else base
    extra = [] if extra is None else extra
    base["extra"] = extra
    return base


def _bad_merge(base: Dict[str, Any] = {}, extra: List[int] = []):
    """Buggy version kept only to demonstrate the failure mode."""
    base["extra"] = extra
    return base


def test_mutable_defaults_are_isolated_between_calls():
    first = _good_merge()
    first["extra"].append(1)
    second = _good_merge()
    assert second == {"extra": []}
    assert first == {"extra": [1]}


def test_buggy_mutable_default_leaks_state():
    # Sanity check that the pattern we are guarding against is real.
    _bad_merge()["extra"].append(1)
    assert _bad_merge()["extra"] == [1]


# ---------------------------------------------------------------------------
# 2. Module-level dict mutated in place
# ---------------------------------------------------------------------------

_FALLBACK_SHARDS = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _resolve_shards(proj_name: str, fused_mapping: Optional[Dict[str, List[str]]] = None):
    """Fixed version: copy the module-level mapping before mutating."""
    effective = dict(_FALLBACK_SHARDS)
    if fused_mapping is not None:
        effective.update(fused_mapping)
    return list(effective.get(proj_name, []))


def test_module_level_mapping_is_not_polluted():
    before = {k: list(v) for k, v in _FALLBACK_SHARDS.items()}
    _resolve_shards("qkv_proj")
    _resolve_shards("gate_up_proj", {"gate_up_proj": ["gate_proj"]})
    assert {k: list(v) for k, v in _FALLBACK_SHARDS.items()} == before


def test_module_level_mapping_accepts_none():
    assert _resolve_shards("qkv_proj", None) == ["q_proj", "k_proj", "v_proj"]


# ---------------------------------------------------------------------------
# 3. dataclass optional field: hasattr vs is not None
# ---------------------------------------------------------------------------


@dataclass
class _ModelConfig:
    vocab_embedding: Optional[Any] = None
    lm_head: Optional[Any] = None


def _pad_embedding_lm_head_buggy(config: _ModelConfig):
    # hasattr is always True for dataclass fields, so this crashes on None.
    if hasattr(config, "vocab_embedding"):
        return config.vocab_embedding.shape
    return None


def _pad_embedding_lm_head_fixed(config: _ModelConfig):
    if config.vocab_embedding is not None:
        return config.vocab_embedding.shape
    return None


def test_dataclass_hasattr_is_always_true():
    cfg = _ModelConfig()
    assert hasattr(cfg, "vocab_embedding")
    with pytest.raises(AttributeError):
        _pad_embedding_lm_head_buggy(cfg)


def test_dataclass_is_not_none_check_handles_missing_field():
    cfg = _ModelConfig()
    assert _pad_embedding_lm_head_fixed(cfg) is None


# ---------------------------------------------------------------------------
# 4. Unsafe pickle loading
# ---------------------------------------------------------------------------


class _UnsafePayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        # Would create a file if unpickled.
        return (open, (str(self.marker), "w"))


def _load_weights(path, expected_sha256: Optional[str] = None):
    """Fixed loader: verify checksum before unpickling."""
    data = path.read_bytes()
    if expected_sha256 is not None:
        actual = hashlib.sha256(data).hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise RuntimeError("Checkpoint checksum mismatch")
    return pickle.loads(data)


def test_unsafe_pickle_is_rejected_without_execution(tmp_path):
    marker = tmp_path / "executed"
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps(_UnsafePayload(marker)))
    with pytest.raises(RuntimeError, match="checksum"):
        _load_weights(payload, expected_sha256="0" * 64)
    assert not marker.exists()


def test_trusted_pickle_loads_with_matching_checksum(tmp_path):
    payload = tmp_path / "payload.pkl"
    payload.write_bytes(pickle.dumps({"ok": True}))
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    assert _load_weights(payload, expected_sha256=digest) == {"ok": True}


# ---------------------------------------------------------------------------
# 5. MoE init keyword mismatch
# ---------------------------------------------------------------------------


def _init_moe_module(*, orig_router_weights, new_router_weights,
                     orig_num_experts, new_num_experts):
    return {
        "orig_router_weights": orig_router_weights,
        "new_router_weights": new_router_weights,
        "orig_num_experts": orig_num_experts,
        "new_num_experts": new_num_experts,
    }


def test_old_moe_keywords_raise_type_error():
    with pytest.raises(TypeError):
        _init_moe_module(
            orig_router_weight=None,
            new_router_weight=None,
            orig_num_experts=4,
            new_num_experts=2,
        )


def test_new_moe_keywords_are_accepted():
    result = _init_moe_module(
        orig_router_weights=[1, 2, 3, 4],
        new_router_weights=[1, 3],
        orig_num_experts=4,
        new_num_experts=2,
    )
    assert result["orig_num_experts"] == 4
    assert result["new_num_experts"] == 2
    assert result["new_router_weights"] == [1, 3]
