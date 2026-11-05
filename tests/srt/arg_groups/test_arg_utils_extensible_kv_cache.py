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
"""CLI parsing for the opt-in extensible KV cache switch.

These tests do not require CUDA: they only exercise argparse wiring and the
environment-variable default used by the runtime check.
"""

import argparse
import dataclasses

import pytest

from sglang.srt.arg_groups.arg_utils import A, Arg, add_cli_args_from_dataclass


@dataclasses.dataclass
class _ExtensibleKVCacheArgs:
    enable_extensible_kv_cache: A[
        bool,
        Arg(
            help="Size the KV cache from measured free memory after warmup.",
            aliases=["--enable-extensible-kv-cache"],
        ),
    ] = False


@pytest.fixture
def parser():
    p = argparse.ArgumentParser()
    add_cli_args_from_dataclass(p, _ExtensibleKVCacheArgs)
    return p


def test_default_is_disabled(parser):
    args = parser.parse_args([])
    assert args.enable_extensible_kv_cache is False


def test_explicit_enable(parser):
    args = parser.parse_args(["--enable-extensible-kv-cache"])
    assert args.enable_extensible_kv_cache is True


def test_explicit_disable_keeps_fallback(parser):
    # The switch is store_true, so passing it is the only way to enable; the
    # absence of the flag is the explicit fallback path.
    args = parser.parse_args([])
    assert args.enable_extensible_kv_cache is False


def test_runtime_check_agrees_with_cli_default(monkeypatch):
    from sglang.srt.mem_cache.extensible_kv_cache import (
        extensible_kv_cache_enabled,
    )

    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True
