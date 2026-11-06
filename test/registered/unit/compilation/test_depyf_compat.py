# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the depyf ``load_by_key_path`` compatibility patch."""

import inspect
import sys

import pytest

pytest.importorskip("depyf")

import depyf.explain.enable_debugging as enable_debugging
import torch._inductor.codecache as codecache

from sglang.srt.compilation import depyf_compat
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

PRISTINE_HOOK = enable_debugging.patched_load_by_key_path
_SUPPORTS_SET_SYS_MODULES = (
    "set_sys_modules"
    in inspect.signature(codecache.PyCodeCache.load_by_key_path).parameters
)


@pytest.mark.skipif(
    not _SUPPORTS_SET_SYS_MODULES,
    reason="torch Inductor does not pass set_sys_modules in this version",
)
class TestPatchedHook:
    def test_load_by_key_path_accepts_set_sys_modules(self, tmp_path):
        with depyf_compat.prepare_debug(str(tmp_path)):
            key, path = codecache.PyCodeCache.write("value = 1\n")
            module = codecache.PyCodeCache.load_by_key_path(
                key, path, set_sys_modules=False
            )

        assert module.value == 1

    def test_set_sys_modules_is_forwarded_not_swallowed(self, tmp_path):
        with depyf_compat.prepare_debug(str(tmp_path)):
            key, path = codecache.PyCodeCache.write("value = 2\n")
            module = codecache.PyCodeCache.load_by_key_path(
                key, path, set_sys_modules=False
            )
            registered = module.__name__ in sys.modules

        sys.modules.pop(module.__name__, None)
        assert not registered

    def test_depyf_still_dumps_the_generated_kernel(self, tmp_path):
        with depyf_compat.prepare_debug(str(tmp_path)):
            key, path = codecache.PyCodeCache.write("value = 3\n")
            codecache.PyCodeCache.load_by_key_path(
                key, path, set_sys_modules=False
            )

        dumped = list(tmp_path.rglob("*.kernel_*.py"))
        assert dumped, f"depyf wrote no kernel dump into {tmp_path}"
        assert "value = 3" in dumped[0].read_text()


class TestPatchApplication:
    def test_patch_is_skipped_when_unnecessary(self):
        def accepts_anything(*args, **kwargs):
            pass

        assert depyf_compat._hook_accepts_inductor_keywords(accepts_anything)

    @pytest.mark.skipif(
        not _SUPPORTS_SET_SYS_MODULES,
        reason="torch Inductor does not pass set_sys_modules in this version",
    )
    def test_patch_is_still_needed(self):
        assert not depyf_compat._hook_accepts_inductor_keywords(PRISTINE_HOOK), (
            "depyf's load_by_key_path now accepts everything Inductor passes; "
            "drop depyf_compat and this test"
        )

    def test_patch_is_idempotent(self, tmp_path):
        with depyf_compat.prepare_debug(str(tmp_path)):
            first = enable_debugging.patched_load_by_key_path
        with depyf_compat.prepare_debug(str(tmp_path)):
            second = enable_debugging.patched_load_by_key_path

        assert first is second
