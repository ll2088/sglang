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

from sglang.srt.arg_groups.arg_utils import _env_flag_default


def test_env_flag_default_off(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert _env_flag_default("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE") is False


def test_env_flag_default_on(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert _env_flag_default("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE") is True


def test_env_flag_default_explicit_off(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "0")
    assert _env_flag_default("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE") is False
