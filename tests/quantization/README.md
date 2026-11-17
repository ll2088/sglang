# Activation-quantization fallback tests

This directory is intentionally **self-contained**. It verifies the explicit
activation-quantization fallback handling and the `VLLM_STRICT_QUANT_SCHEME`
strict mode without importing the `sglang` package.

## Why self-contained

The module under test
(`python/sglang/srt/layers/quantization/activation_fallback.py`) only uses the
standard library. The test module loads it directly from its file path, so no
`sglang` install is required. This keeps the suite runnable on GPU-less CI
runners where `pip install -e python` would pull the full runtime dependency
set (py-spy, pydantic, quack-kernels, ...) and fail during environment setup.

## How to run

Install only pytest, then run with the directory-local config:

```bash
pip install -r tests/quantization/requirements.txt
python -m pytest -q -c tests/quantization/pytest.ini tests/quantization
```

Do **not** run `pip install -e python` (or `pip install sglang`) before this
suite. The `-c tests/quantization/pytest.ini` flag pins rootdir/testpaths to
this directory so the repository-root pytest configuration is not picked up.

## What is covered

- `VLLM_STRICT_QUANT_SCHEME` parsing (`""`/`"0"` disabled, anything else on).
- `executed_activation_key` dispatch, including per-backend declarations
  (humming / marlin / mxfp4 / fp8).
- `check_activation_quant_fallback` for matching, mismatching, and
  nothing-requested cases, in both default and strict mode.
- Backend selection: default mode picks the fallback kernel, strict mode
  rejects it while preserving priority order so a later honoring kernel can
  still be selected.
- MoE runner declarations (base / humming / marlin).
