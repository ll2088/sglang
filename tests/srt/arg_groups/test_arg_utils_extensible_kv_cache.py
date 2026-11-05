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

These tests do not touch CUDA: they only exercise the argparse plumbing and the
environment-variable default so the switch can be turned on and off without
changing code.
"""

import argparse
import dataclasses
import os
from typing import Annotated

import pytest

from sglang.srt.arg_groups.arg_utils import (
    Arg,
    _env_flag_default,
    add_cli_args_from_dataclass,
)

ENV_NAME = "SGLANG_ENABLE_EXTENSIBLE_KV_CACHE"


@dataclasses.dataclass
class _ExtensibleKVCacheArgs:
    enable_extensible_kv_cache: Annotated[
        bool,
        Arg(
            help="Size the KV cache from measured free memory after warmup.",
            aliases=["--extensible-kv-cache"],
        ),
    ] = _env_flag_default(ENV_NAME, default=False)


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_cli_args_from_dataclass(parser, _ExtensibleKVCacheArgs)
    return parser.parse_args(argv)


def test_default_is_disabled(monkeypatch):
    monkeypatch.delenv(ENV_NAME, raising=False)
    args = _parse([])
    assert args.enable_extensible_kv_cache is False


def test_explicit_enable():
    args = _parse(["--enable-extensible-kv-cache"])
    assert args.enable_extensible_kv_cache is True


def test_alias_enable():
    args = _parse(["--extensible-kv-cache"])
    assert args.enable_extensible_kv_cache is True


def test_env_default_enables(monkeypatch):
    monkeypatch.setenv(ENV_NAME, "1")
    # The dataclass default is evaluated at import time, so re-read the helper
    # to confirm the env flag semantics the runtime check relies on.
    assert _env_flag_default(ENV_NAME, default=False) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_env_default_disabled_values(monkeypatch, value):
    monkeypatch.setenv(ENV_NAME, value)
    assert _env_flag_default(ENV_NAME, default=False) is False


def test_runtime_flag_matches_env(monkeypatch):
    from sglang.srt.mem_cache.extensible_kv_cache import (
        extensible_kv_cache_enabled,
    )

    monkeypatch.delenv(ENV_NAME, raising=False)
    assert extensible_kv_cache_enabled() is False
    monkeypatch.setenv(ENV_NAME, "1")
    assert extensible_kv_cache_enabled() is True
    monkeypatch.setenv(ENV_NAME, "0")
    assert extensible_kv_cache_enabled() is False
