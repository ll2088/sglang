# Copyright 2023-2024 SGLang Team
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
"""Keep the OpenAI unit tests importable without the full sglang runtime.

The CI setup step runs ``pip install -e python``, which pulls heavy runtime
dependencies (torch, quack-kernels, ...). These unit tests only need pydantic
and pytest, so make sure a partially installed ``sglang`` package never leaks
into collection: the test modules load the sources they need explicitly.
"""

import sys
import types


def _install_package_stubs() -> None:
    """Pre-seed ``sglang`` package stubs before any test module imports.

    The CI setup step runs ``pip install -e python`` which resolves the full
    runtime dependency chain (torch, quack-kernels, ...). These unit tests only
    need pydantic + pytest, so install empty package stubs at conftest import
    time (before collection) to keep the real ``sglang.__init__`` from ever
    being imported.
    """
    for pkg_name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.entrypoints",
        "sglang.srt.entrypoints.openai",
    ):
        existing = sys.modules.get(pkg_name)
        if existing is not None and getattr(existing, "__file__", None) is not None:
            # A real package was already imported; leave it alone.
            continue
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = []
        sys.modules[pkg_name] = pkg


_install_package_stubs()


def pytest_configure(config):
    # Re-assert stubs in case a plugin imported the real package first.
    _install_package_stubs()
