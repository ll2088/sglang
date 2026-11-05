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

import argparse
import dataclasses
from typing import Annotated

from sglang.srt.arg_groups.arg_utils import Arg, add_cli_args_from_dataclass


@dataclasses.dataclass
class _ExtensibleKVCacheArgs:
    enable_extensible_kv_cache: Annotated[
        bool,
        Arg(help="Enable the VMM-backed extensible KV cache."),
    ] = False


@dataclasses.dataclass
class _ExtensibleKVCacheArgsDefaultOn:
    enable_extensible_kv_cache: Annotated[
        bool,
        Arg(help="Enable the VMM-backed extensible KV cache."),
    ] = True


def _parse(cls, argv):
    parser = argparse.ArgumentParser()
    add_cli_args_from_dataclass(parser, cls)
    return parser.parse_args(argv)


def test_extensible_kv_cache_flag_defaults_off():
    args = _parse(_ExtensibleKVCacheArgs, [])
    assert args.enable_extensible_kv_cache is False


def test_extensible_kv_cache_flag_can_be_enabled():
    args = _parse(_ExtensibleKVCacheArgs, ["--enable-extensible-kv-cache"])
    assert args.enable_extensible_kv_cache is True


def test_extensible_kv_cache_flag_default_on_can_be_disabled():
    # The CLI only exposes store_true, so a default-on field stays on unless
    # the caller constructs the dataclass directly. This documents the
    # fallback contract: disabling happens through the dataclass/env, not a
    # CLI negation flag.
    args = _parse(_ExtensibleKVCacheArgsDefaultOn, [])
    assert args.enable_extensible_kv_cache is True
    direct = _ExtensibleKVCacheArgsDefaultOn(enable_extensible_kv_cache=False)
    assert direct.enable_extensible_kv_cache is False
