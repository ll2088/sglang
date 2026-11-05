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
"""CLI parsing tests for the extensible KV cache switch."""

import argparse
import dataclasses
import os
from typing import Optional
from unittest import mock

from sglang.srt.arg_groups.arg_utils import A, Arg, add_cli_args_from_dataclass


@dataclasses.dataclass
class _ExtensibleKVCacheArgs:
    enable_extensible_kv_cache: A[
        bool, Arg(help="Enable the VMM-backed extensible KV cache.")
    ] = False
    extensible_kv_cache_max_blocks: A[
        Optional[int], Arg(help="Upper bound on reserved KV cache blocks.")
    ] = None


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_cli_args_from_dataclass(parser, _ExtensibleKVCacheArgs)
    return parser.parse_args(argv)


def test_default_is_disabled():
    args = _parse([])
    assert args.enable_extensible_kv_cache is False
    assert args.extensible_kv_cache_max_blocks is None


def test_explicit_enable():
    args = _parse(["--enable-extensible-kv-cache"])
    assert args.enable_extensible_kv_cache is True


def test_explicit_max_blocks():
    args = _parse(["--extensible-kv-cache-max-blocks", "128"])
    assert args.extensible_kv_cache_max_blocks == 128


def test_runtime_flag_defaults_off():
    from sglang.srt.mem_cache.extensible_kv_cache import (
        extensible_kv_cache_enabled,
    )

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", None)
        assert extensible_kv_cache_enabled() is False


def test_runtime_flag_accepts_truthy_values():
    from sglang.srt.mem_cache.extensible_kv_cache import (
        extensible_kv_cache_enabled,
    )

    for value in ("1", "true", "TRUE", "yes", "on"):
        with mock.patch.dict(
            os.environ, {"SGLANG_ENABLE_EXTENSIBLE_KV_CACHE": value}
        ):
            assert extensible_kv_cache_enabled() is True


def test_runtime_flag_rejects_falsey_values():
    from sglang.srt.mem_cache.extensible_kv_cache import (
        extensible_kv_cache_enabled,
    )

    for value in ("0", "false", "no", "off", ""):
        with mock.patch.dict(
            os.environ, {"SGLANG_ENABLE_EXTENSIBLE_KV_CACHE": value}
        ):
            assert extensible_kv_cache_enabled() is False
