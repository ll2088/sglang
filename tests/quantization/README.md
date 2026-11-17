# Activation-quantization fallback suite

This directory contains the self-contained test suite for
`python/sglang/srt/layers/quantization/activation_fallback.py`, which makes
activation-quantization fallbacks explicit and adds the
`VLLM_STRICT_QUANT_SCHEME` strict-mode switch.

## Why this suite is self-contained

The module under test only uses the Python standard library. Importing
`sglang` (or any of its submodules) resolves the full runtime dependency set
(`py-spy`, `pybase64`, `pydantic`, `quack-kernels`, ...), which is what made
CI environment preparation fail. To keep that from happening, this suite:

* declares only `pytest` in `requirements.txt`,
* strips the repository root and its `python/` directory from `sys.path` and
  `PYTHONPATH` in `conftest.py`,
* loads `activation_fallback.py` directly from its source file instead of
  importing it through the `sglang` package,
* ships its own `pytest.ini` so pytest does not pick up the repository-root
  configuration.

## Running the suite

Install only the minimal requirements and run pytest with the explicit
config file:

```bash
pip install -r tests/quantization/requirements.txt
python -m pytest -q -c tests/quantization/pytest.ini \
    tests/quantization/test_activation_fallback.py
```

Do **not** run `pip install -e python` for this suite, and do not invoke
pytest without `-c tests/quantization/pytest.ini`: either would resolve the
full sglang dependency set during environment preparation.

## What is covered

* `is_strict_quant_scheme()` reading `VLLM_STRICT_QUANT_SCHEME` lazily.
* `executed_activation_key()` defaulting to the requested key and honoring
  per-kernel declarations.
* `check_activation_quant_fallback()` returning a reason in default mode and
  raising in strict mode, with a message naming the requested and executed
  schemes.
* `check_runner_activation_fallback()` for the MoE runner path.
* Backend-specific declarations for humming, marlin, mxfp4 and fp8.
