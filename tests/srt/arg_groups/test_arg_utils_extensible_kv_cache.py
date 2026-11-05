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
"""CLI parsing tests for the extensible KV cache opt-in switch.

The switch is read from the environment by ``extensible_kv_cache_enabled``;
this test pins the default (off) and the explicit enable/disable behaviour so
that the fallback path stays the default and cannot be flipped accidentally.
"""

import pytest

from sglang.srt.mem_cache.extensible_kv_cache import extensible_kv_cache_enabled


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("0", False),
        ("1", True),
    ],
)
def test_extensible_kv_cache_env_switch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    else:
        monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", value)
    assert extensible_kv_cache_enabled() is expected


def test_extensible_kv_cache_default_is_off(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False
