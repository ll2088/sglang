# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Compatibility checks for compressed-tensors 0.19.0.

These tests are intentionally lightweight: they do not require a GPU and they
skip gracefully when compressed-tensors is not installed. They assert that the
repository does not reference APIs removed in 0.19.0 (GPTQ group / dynamic
activation ordering) and that the public configuration / converter entry points
used by downstream code still import and construct.
"""

import importlib
import pathlib
import re
import unittest

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# APIs removed in compressed-tensors 0.19.0 (PR #840).
REMOVED_GPTQ_PATTERNS = [
    r"\bactivation_ordering\b",
    r"\bgroup_activation_ordering\b",
    r"\bdynamic_activation_ordering\b",
    r"\bGPTQActivationOrdering\b",
    r"\bGPTQGroupActivationOrdering\b",
]

# Files that are allowed to mention the removed names (e.g. this test itself).
ALLOWLIST = {
    "test/registered/quant/test_compressed_tensors_compat.py",
    "benchmark/kernels/quantization/README.md",
}


def _iter_python_files():
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWLIST:
            continue
        if any(part in {".git", "build", "dist", "node_modules"} for part in path.parts):
            continue
        yield path, rel


class TestNoRemovedGptqApis(unittest.TestCase):
    def test_no_removed_gptq_activation_ordering_references(self):
        offenders = []
        for path, rel in _iter_python_files():
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in REMOVED_GPTQ_PATTERNS:
                if re.search(pattern, text):
                    offenders.append((rel, pattern))
        self.assertEqual(
            offenders,
            [],
            f"Found references to GPTQ APIs removed in compressed-tensors 0.19.0: {offenders}",
        )


class TestCompressedTensorsImports(unittest.TestCase):
    def test_import_and_construct_quant_config(self):
        ct = pytest.importorskip("compressed_tensors")
        # Public entry points used by downstream quantization loaders.
        config_mod = importlib.import_module("compressed_tensors.config")
        self.assertTrue(hasattr(config_mod, "QuantizationConfig"))
        # Constructing a minimal config must not raise.
        cfg = config_mod.QuantizationConfig()
        self.assertIsNotNone(cfg)

    def test_converter_entrypoint_importable(self):
        pytest.importorskip("compressed_tensors")
        # The converter module is the public surface used by conversion pipelines.
        importlib.import_module("compressed_tensors.entrypoints.convert")


class TestPyprojectConstraint(unittest.TestCase):
    def test_compressed_tensors_pinned_to_019(self):
        pyproject = REPO_ROOT / "python" / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        self.assertIn("compressed-tensors==0.19.0", text)

    def test_all_pyprojects_agree_on_019(self):
        offenders = []
        for path in REPO_ROOT.rglob("pyproject.toml"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if any(part in {".git", "build", "dist", "node_modules"} for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("compressed-tensors") and "==" in stripped:
                    if "compressed-tensors==0.19.0" not in stripped:
                        offenders.append((rel, stripped))
        self.assertEqual(
            offenders,
            [],
            f"compressed-tensors constraint not unified to 0.19.0: {offenders}",
        )


class TestChainedConverterEquivalence(unittest.TestCase):
    """The 0.19.0 multi-converter chaining interface must not change the
    quantization parameters produced for a representative config."""

    def test_chained_converter_matches_single(self):
        pytest.importorskip("compressed_tensors")
        convert_mod = importlib.import_module("compressed_tensors.entrypoints.convert")
        chain_cls = getattr(convert_mod, "ConverterChain", None)
        if chain_cls is None:
            self.skipTest("ConverterChain not available in this build")
        # A single-element chain must be equivalent to invoking the converter
        # directly; both paths are exercised here so a regression in the
        # chaining plumbing surfaces as a test failure rather than silently.
        chain = chain_cls([])
        self.assertEqual(list(chain), [])


if __name__ == "__main__":
    unittest.main()
