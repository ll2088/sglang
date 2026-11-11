"""Make the diffusion perf fixture reachable from this directory.

``DiffusionServerBase`` (subclassed by the diffusion tests here) has an autouse
fixture that requests ``perf_results``, which is declared in
``sglang/multimodal_gen/test/server/conftest.py``. pytest resolves conftest.py
by the collected file's directory, so that declaration never reaches
``test/registered/``; re-exporting it here is what makes it resolvable.
"""

from sglang.multimodal_gen.test.server.conftest import perf_results  # noqa: F401

import os

import pytest

# Backend switch matrix for AMD/ROCm tests. Each entry maps environment
# variable names to the values that select that configuration. Keep this
# table small and explicit so failures point at a single switch.
AITER_CONFIGS = {
    "aiter_off": {"SGLANG_USE_AITER": "0"},
    "aiter_on": {"SGLANG_USE_AITER": "1"},
}


def _refresh_backend_envs():
    """Drop any cached view of the backend env vars.

    SGLang reads backend switches through ``os.environ`` at import/init time.
    Tests that flip them must invalidate whatever cache exists so the next
    test does not observe the previous configuration. This helper is a no-op
    when no cache is present, which keeps it safe across versions and when the
    sglang package is not importable (e.g. a minimal CI image).
    """
    try:
        from sglang.srt.utils import envs as sglang_envs
    except Exception:
        return
    disable = getattr(sglang_envs, "disable_envs_cache", None)
    if callable(disable):
        try:
            disable()
        except Exception:
            pass


def _aiter_supported() -> bool:
    """Best-effort detection of an AITER-capable ROCm environment."""
    if os.environ.get("SGLANG_USE_AITER") == "1":
        return True
    try:
        import torch
    except Exception:
        return False
    return bool(getattr(getattr(torch, "version", None), "hip", None))


@pytest.fixture(
    params=list(AITER_CONFIGS) if _aiter_supported() else ["aiter_off"]
)
def aiter_config(request):
    """Parameterize tests over the AITER on/off matrix.

    The fixture sets the env vars for the selected configuration, refreshes
    any cached backend state, yields the configuration name, and restores the
    previous environment afterwards so repeated or reordered runs do not leak
    state between tests.
    """
    with pytest.MonkeyPatch.context() as mp:
        for name, value in AITER_CONFIGS[request.param].items():
            mp.setenv(name, value)
        _refresh_backend_envs()
        yield request.param
    _refresh_backend_envs()
