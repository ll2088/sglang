# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/models/registry.py

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import AbstractSet, Dict, List, Optional, Tuple, Type, Union

import torch.nn as nn

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _parse_version(version: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _get_transformers_version() -> Optional[str]:
    try:
        import transformers
    except Exception:
        return None
    return getattr(transformers, "__version__", None)


def check_min_transformers_version(
    arch: str, min_version: str, reason: Optional[str] = None
) -> None:
    """Raise a readable ImportError when the installed transformers is too old."""
    installed = _get_transformers_version()
    if installed is None:
        return
    if _parse_version(installed) >= _parse_version(min_version):
        return
    detail = f" ({reason})" if reason else ""
    raise ImportError(
        f"Model architecture '{arch}' requires transformers>={min_version}, "
        f"but transformers {installed} is installed{detail}. "
        f"Please upgrade transformers: pip install -U 'transformers>={min_version}'."
    )


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: Dict[str, Union[Type[nn.Module], str]] = field(default_factory=dict)
    # Optional minimum transformers version per model_arch, plus a human
    # readable reason shown in the ImportError when the gate is not met.
    min_transformers_versions: Dict[str, str] = field(default_factory=dict)
    transformers_version_reasons: Dict[str, str] = field(default_factory=dict)
    _default_models_loaded: bool = True

    def set_min_transformers_version(
        self, model_arch: str, min_version: str, reason: Optional[str] = None
    ) -> None:
        self.min_transformers_versions[model_arch] = min_version
        if reason:
            self.transformers_version_reasons[model_arch] = reason

    def _load_default_models(self):
        if self._default_models_loaded:
            return

        self.models = import_model_classes("sglang.srt.models") | self.models
        self._default_models_loaded = True

    def register(
        self, package_name: str, overwrite: bool = False, strict: bool = False
    ):
        self._load_default_models()
        new_models = import_model_classes(package_name, strict=strict)
        if overwrite:
            self.models.update(new_models)
        else:
            for arch, cls in new_models.items():
                if arch in self.models:
                    raise ValueError(
                        f"Model architecture {arch} already registered. Set overwrite=True to replace."
                    )
                self.models[arch] = cls

    def get_supported_archs(self) -> AbstractSet[str]:
        self._load_default_models()
        return self.models.keys()

    def is_arch_supported(self, architecture: str) -> bool:
        return architecture in self.models or architecture in self.get_supported_archs()

    def _raise_for_unsupported(self, architectures: List[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed "
                "to be inspected. Please check the logs for more details."
            )

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )

    def _try_load_model_cls(self, model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch not in self.models:
            return None

        min_version = self.min_transformers_versions.get(model_arch)
        if min_version is not None:
            check_min_transformers_version(
                model_arch,
                min_version,
                self.transformers_version_reasons.get(model_arch),
            )

        return self.models[model_arch]

    def _normalize_archs(
        self,
        architectures: Union[str, List[str]],
    ) -> List[str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            logger.warning("No model architectures are specified")

        # filter out support architectures
        normalized_arch = list(
            filter(lambda model: model in self.models, architectures)
        )

        # make sure Transformers backend is put at the last as a fallback
        if len(normalized_arch) != len(architectures):
            normalized_arch.append("TransformersForCausalLM")
        return normalized_arch

    def resolve_model_cls(
        self,
        architectures: Union[str, List[str]],
    ) -> Tuple[Type[nn.Module], str]:
        # Fast startup path: resolve registered models without importing defaults.
        if architectures:
            first_arch = (
                architectures if isinstance(architectures, str) else architectures[0]
            )
            model_cls = self.models.get(first_arch)
            if model_cls is not None:
                return model_cls, first_arch

        self._load_default_models()
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        return self._raise_for_unsupported(architectures)


@lru_cache()
def import_model_classes(package_name: str, strict: bool = False):
    model_arch_name_to_cls = {}
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            if name.split(".")[-1] in envs.SGLANG_DISABLED_MODEL_ARCHS.get():
                logger.debug(f"Skip loading {name} due to SGLANG_DISABLED_MODEL_ARCHS")
                continue

            try:
                module = importlib.import_module(name)
            except Exception as e:
                if strict:
                    raise
                logger.warning(f"Ignore import error when loading {name}: {e}")
                continue
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(
                    entry, list
                ):  # To support multiple model classes in one module
                    for tmp in entry:
                        assert tmp.__name__ not in model_arch_name_to_cls, (
                            f"Duplicated model implementation for {tmp.__name__}"
                        )
                        model_arch_name_to_cls[tmp.__name__] = tmp
                else:
                    assert entry.__name__ not in model_arch_name_to_cls, (
                        f"Duplicated model implementation for {entry.__name__}"
                    )
                    model_arch_name_to_cls[entry.__name__] = entry

    return model_arch_name_to_cls


ModelRegistry = _ModelRegistry()
# Models that are only usable through the Transformers modeling backend need a
# minimum transformers release. Declaring it here keeps the failure mode a
# readable ImportError instead of an obscure AttributeError at load time.
ModelRegistry.set_min_transformers_version(
    "AriaForConditionalGeneration",
    "5.18.0",
    "Needs https://github.com/huggingface/transformers/pull/48907",
)
if external_pkg := envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get():
    # Defer default model imports to speed up startup when using external models.
    ModelRegistry._default_models_loaded = False
    ModelRegistry.models.update(import_model_classes(external_pkg))
else:
    ModelRegistry.register("sglang.srt.models")
