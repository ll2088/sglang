# tests/quantization

Self-contained tests for explicit activation-quantization fallback handling
(`python/sglang/srt/layers/quantization/activation_fallback.py`).

## Why this directory is self-contained

These tests must run on GPU-less CI runners. They load the module under test
directly from its file path and only use the standard library plus `pytest`.
They must **not** require `pip install -e python` (or `pip install sglang`),
which pulls the full sglang runtime dependency set and fails on such runners.

## How to run

Install only the minimal dependencies and run with the directory-local
`pytest.ini` so the repository-root configuration is not picked up:

```bash
pip install -r tests/quantization/requirements.txt
python -m pytest -q -c tests/quantization/pytest.ini tests/quantization
```

Do **not** run `pip install -e python` before this suite.

## What is covered

- `VLLM_STRICT_QUANT_SCHEME` parsing (`""`/`"0"` disabled, anything else enabled).
- `executed_activation_key` dispatch for kernels that honor, declare, or drop
  the requested activation quantization (humming / marlin / mxfp4 / fp8).
- `check_activation_quant_fallback` in default mode (once-per-kernel warning,
  returns a reason) and strict mode (raises `ValueError` mentioning
  `VLLM_STRICT_QUANT_SCHEME` and the requested/executed mismatch).
- Backend selection loop: default mode selects the fallback kernel, strict mode
  skips it and falls through to a later kernel that honors the scheme, or fails
  with the mismatch reason.
- MoE runner declarations (`MoeRunnerCore` base, humming, marlin).
