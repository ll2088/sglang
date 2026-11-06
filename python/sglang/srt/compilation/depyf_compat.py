# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep depyf usable against Inductor versions newer than its last release.

``depyf.prepare_debug`` replaces several torch functions with hooks of its
own. ``load_by_key_path`` has a frozen signature that predates Inductor's
``set_sys_modules`` argument, so autotune code blocks fail while depyf is
active. Patch the hook here, following depyf's own ``lazy_format_graph_code``
approach: accept ``**kwargs`` and forward them.

Route every depyf entry point through ``prepare_debug`` below so the patch is
never missed.
"""

import inspect
import logging
import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

logger = logging.getLogger(__name__)

_patched = False


def _hook_accepts_inductor_keywords(hook: Callable[..., Any]) -> bool:
    """Whether depyf's hook already takes everything Inductor passes it."""
    import torch._inductor.codecache as codecache

    accepted = inspect.signature(hook).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        return True
    expected = inspect.signature(codecache.PyCodeCache.load_by_key_path).parameters
    return expected.keys() <= accepted.keys()


def _patch_load_by_key_path() -> None:
    """Forward unrecognised keywords through depyf's hook."""
    import depyf.explain.enable_debugging as enable_debugging
    from depyf.explain.global_variables import data
    from depyf.explain.utils import (
        get_current_compiled_fn_name,
        write_code_to_file_template,
    )

    def load_by_key_path(key, path, linemap=None, attrs=None, **kwargs):
        with open(path) as f:
            src = f.read()
        func_name = get_current_compiled_fn_name()
        dumped_path = write_code_to_file_template(
            src, os.path.join(data["dump_src_dir"], func_name + ".kernel_%s.py")
        )
        return data["unpatched_load_by_key_path"](
            key, dumped_path, linemap, attrs, **kwargs
        )

    enable_debugging.patched_load_by_key_path = load_by_key_path


def prepare_debug(dump_src_dir: str) -> AbstractContextManager[None]:
    """``depyf.prepare_debug`` with the compatibility patch applied."""
    global _patched
    try:
        import depyf
        import depyf.explain.enable_debugging as enable_debugging
    except ImportError:
        logger.warning("depyf is not installed; debug dump is disabled")
        return nullcontext()

    if not _patched:
        if not _hook_accepts_inductor_keywords(
            enable_debugging.patched_load_by_key_path
        ):
            logger.debug("Patching depyf's load_by_key_path hook for this Inductor")
            _patch_load_by_key_path()
        _patched = True

    return depyf.prepare_debug(dump_src_dir)
