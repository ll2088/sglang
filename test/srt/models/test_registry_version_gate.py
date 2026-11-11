# SPDX-License-Identifier: Apache-2.0
"""Tests for the min_transformers_version gate in the model registries."""

import pytest

from sglang.srt.models import registry as srt_registry


class _DummyModel:
    pass


def test_aria_registered_with_version_gate():
    registry = srt_registry._ModelRegistry()
    registry.models["AriaForConditionalGeneration"] = _DummyModel
    registry.set_min_transformers_version(
        "AriaForConditionalGeneration",
        "5.18.0",
        "Needs https://github.com/huggingface/transformers/pull/48907",
    )

    assert registry.min_transformers_versions["AriaForConditionalGeneration"] == "5.18.0"
    assert "48907" in registry.transformers_version_reasons[
        "AriaForConditionalGeneration"
    ]


def test_version_gate_passes_when_new_enough(monkeypatch):
    monkeypatch.setattr(
        srt_registry, "_get_transformers_version", lambda: "5.19.0"
    )
    registry = srt_registry._ModelRegistry()
    registry.models["AriaForConditionalGeneration"] = _DummyModel
    registry.set_min_transformers_version("AriaForConditionalGeneration", "5.18.0")

    assert registry._try_load_model_cls("AriaForConditionalGeneration") is _DummyModel


def test_version_gate_raises_readable_error_when_too_old(monkeypatch):
    monkeypatch.setattr(
        srt_registry, "_get_transformers_version", lambda: "5.10.0"
    )
    registry = srt_registry._ModelRegistry()
    registry.models["AriaForConditionalGeneration"] = _DummyModel
    registry.set_min_transformers_version(
        "AriaForConditionalGeneration",
        "5.18.0",
        "Needs https://github.com/huggingface/transformers/pull/48907",
    )

    with pytest.raises(ImportError) as excinfo:
        registry._try_load_model_cls("AriaForConditionalGeneration")
    message = str(excinfo.value)
    assert "AriaForConditionalGeneration" in message
    assert "5.18.0" in message
    assert "5.10.0" in message
    assert "pip install" in message


def test_version_gate_skipped_when_transformers_missing(monkeypatch):
    monkeypatch.setattr(srt_registry, "_get_transformers_version", lambda: None)
    registry = srt_registry._ModelRegistry()
    registry.models["AriaForConditionalGeneration"] = _DummyModel
    registry.set_min_transformers_version("AriaForConditionalGeneration", "5.18.0")

    assert registry._try_load_model_cls("AriaForConditionalGeneration") is _DummyModel


def test_multimodal_gen_registry_has_aria_entry():
    from sglang.multimodal_gen.runtime.models import registry as mm_registry

    assert "AriaForConditionalGeneration" in mm_registry._MIN_TRANSFORMERS_VERSIONS
    assert (
        mm_registry._MIN_TRANSFORMERS_VERSIONS["AriaForConditionalGeneration"]
        == "5.18.0"
    )
    assert "AriaForConditionalGeneration" in mm_registry.ModelRegistry.registered_models
