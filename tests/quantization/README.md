# Activation-quantization fallback tests

This directory is intentionally **self-contained**. It verifies the explicit
activation-quantization fallback handling (`executed_activation_key`,
`check_activation_quant_fallback`, `VLLM_STRICT_QUANT_SCHEME`) without
importing the `sglang` package, so it can run on GPU-less CI runners.

## Dependencies

Only `pytest` is required:

```bash
pip install -r tests/quantization/requirements.txt
```

Do **not** run `pip install -e python` for this suite. The full sglang
runtime dependency set (including heavy packages such as `quack-kernels`)
is not needed and its installation is what previously broke CI setup.

## Running

```bash
python -m pytest -q -c tests/quantization/pytest.ini tests/quantization/test_activation_fallback.py
```

The `-c tests/quantization/pytest.ini` flag pins the rootdir to this
directory so the repository-wide pytest configuration is not picked up.

## What is covered

- `VLLM_STRICT_QUANT_SCHEME=1` rejects kernels that would silently run
  activations unquantized, with an error message naming the mismatch.
- Default mode selects the fallback kernel and emits one warning per kernel.
- `executed_activation_key` declarations for humming / marlin / mxfp4 / fp8
  backends and for the base / humming / marlin MoE runners.
- The unified `check_activation_quant_fallback` helper for valid and invalid
  activation-quantization combinations.
