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

"""CLI parsing for the opt-in extensible KV cache switch.

The switch is read from the environment by
``sglang.srt.mem_cache.extensible_kv_cache.extensible_kv_cache_enabled``.
These tests pin the default (off) and the explicit opt-in/opt-out so a
regression in the env parsing cannot silently enable the VMM path.
"""

import importlib

import pytest


@pytest.fixture
def extensible_module(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    import sglang.srt.mem_cache.extensible_kv_cache as mod

    return importlib.reload(mod)


def test_default_is_disabled(extensible_module):
    assert extensible_module.extensible_kv_cache_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_explicit_enable(monkeypatch, extensible_module, value):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", value)
    assert extensible_module.extensible_kv_cache_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_explicit_disable(monkeypatch, extensible_module, value):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", value)
    assert extensible_module.extensible_kv_cache_enabled() is False


def test_plan_returns_none_when_disabled(extensible_module):
    config = extensible_module.ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20,
        max_num_blocks=4,
        device_index=0,
    )
    assert extensible_module.plan_extensible_kv_cache(config) is None
