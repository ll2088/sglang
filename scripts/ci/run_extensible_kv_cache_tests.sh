#!/usr/bin/env bash
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
# Run the extensible KV cache test suite inside the prepared venv.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
source .venv/bin/activate

python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

TEST_FILES=(
  tests/srt/mem_cache/test_vmm_allocator.py
  tests/srt/mem_cache/test_extensible_kv_cache.py
  tests/srt/arg_groups/test_arg_utils_extensible_kv_cache.py
)

python -m pytest -q -p no:randomly "${TEST_FILES[@]}"
