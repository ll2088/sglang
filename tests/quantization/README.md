# Activation-quantization fallback tests

This directory contains the self-contained test suite for the explicit
activation-quantization fallback handling (`VLLM_STRICT_QUANT_SCHEME`).

## Why this suite is self-contained

The suite loads the module under test directly from its file path
(`python/sglang/srt/layers/quantization/activation_fallback.py`) and only
uses the standard library plus `pytest`. It must **not** require a full
`pip install -e python` of sglang (no GPU, no compiled kernels, no heavy
third-party runtime dependencies).

## How CI should run it

From the repository root:

```bash
pip install -r tests/quantization/requirements.txt
python -m pytest -q tests/quantization/test_activation_fallback.py
```

Do **not** run `pip install -e python` for this directory: that pulls the
full sglang dependency set (py-spy, pybase64, pydantic, quack-kernels, ...)
which is unnecessary here and has been observed to fail during dependency
resolution in CI.

## What is covered

- `VLLM_STRICT_QUANT_SCHEME` parsing (`""`/`"0"` disabled, anything else
  enabled).
- `executed_activation_key` dispatch for kernels that honor the requested
  scheme, weight-only kernels (Marlin, MXFP4), and Humming's input-schema
  dependent behavior.
- `check_activation_quant_fallback` returning `None` on match, a reason in
  default mode, and raising `ValueError` in strict mode.
- Backend selection loop behavior: default mode picks the fallback kernel,
  strict mode rejects it and falls through to a later honoring kernel.
- MoE runner declarations for base / humming / marlin runners.
