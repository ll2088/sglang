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
"""Dependency-constraint checks for compressed-tensors 0.19.0.

These tests are lightweight and do not require a GPU. They assert that every
pyproject.toml in the repository that pins compressed-tensors agrees on
0.19.0, and that the constraint is parseable by packaging.
"""

import pathlib
import unittest

from packaging.requirements import Requirement

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
EXPECTED = "compressed-tensors==0.19.0"


def _iter_pyprojects():
    for path in REPO_ROOT.rglob("pyproject.toml"):
        if any(part in {".git", "build", "dist", "node_modules"} for part in path.parts):
            continue
        yield path


class TestCompressedTensorsConstraint(unittest.TestCase):
    def test_all_pyprojects_agree_on_019(self):
        offenders = []
        for path in _iter_pyprojects():
            rel = path.relative_to(REPO_ROOT).as_posix()
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                stripped = line.strip().strip('"').strip("'").rstrip(",")
                if not stripped.startswith("compressed-tensors"):
                    continue
                if "==" not in stripped:
                    continue
                # Normalize whitespace so "compressed-tensors == 0.19.0" also matches.
                normalized = stripped.replace(" ", "")
                if EXPECTED not in normalized:
                    offenders.append((rel, stripped))
        self.assertEqual(
            offenders,
            [],
            f"compressed-tensors constraint not unified to 0.19.0: {offenders}",
        )

    def test_constraint_is_parseable(self):
        req = Requirement(EXPECTED)
        self.assertEqual(req.name, "compressed-tensors")
        self.assertEqual(str(req.specifier), "==0.19.0")


if __name__ == "__main__":
    unittest.main()
