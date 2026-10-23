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
"""Smoke tests for quantization benchmark scripts.

These tests only verify that the scripts can be imported / parsed without a GPU.
They do not execute the benchmarks themselves.
"""

import ast
import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BENCH_DIR = REPO_ROOT / "benchmark" / "kernels" / "quantization"

SCRIPTS = [
    "bench_int8_quant.py",
    "tuning_block_wise_kernel.py",
    "tuning_mxfp8_native_gfx95.py",
]


class TestQuantBenchScriptsParse(unittest.TestCase):
    def test_scripts_parse(self):
        missing = []
        for name in SCRIPTS:
            path = BENCH_DIR / name
            if not path.exists():
                missing.append(name)
                continue
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
        if missing:
            self.skipTest(f"missing benchmark scripts: {missing}")

    def test_scripts_have_main_guard(self):
        # A smoke runnable script must expose a __main__ guard so CI can invoke
        # it with --help without executing the benchmark body.
        for name in SCRIPTS:
            path = BENCH_DIR / name
            if not path.exists():
                self.skipTest(f"missing benchmark script: {path}")
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            has_guard = any(
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
                for node in tree.body
            )
            self.assertTrue(has_guard, f"{name} lacks a __main__ guard")


if __name__ == "__main__":
    unittest.main()
