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


def pytest_configure(config):
    # If the real package was imported before collection (e.g. by a plugin),
    # leave it alone; the test modules install their own stubs.
    mod = sys.modules.get("sglang")
    if mod is not None and getattr(mod, "__file__", None) is not None:
        return
