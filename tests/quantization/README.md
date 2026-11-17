# Activation-quantization fallback tests

This directory holds the self-contained test suite for explicit
activation-quantization fallback handling (`VLLM_STRICT_QUANT_SCHEME`).

## Why it is self-contained

The suite must run on GPU-less CI runners. It therefore does **not** import
the `sglang` package. Instead, `test_activation_fallback.py` loads the module
under test directly from its file path:

```
python/sglang/srt/layers/quantization/activation_fallback.py
```

That module only uses the standard library, so no compiled kernels and no
third-party runtime dependencies are required.

## How to run

Install only the minimal dependencies and run pytest with the directory-local
configuration:

```bash
pip install -r tests/quantization/requirements.txt
python -m pytest -q -c tests/quantization/pytest.ini tests/quantization
```

Do **not** run `pip install -e python` (or `pip install sglang`) before this
suite. That pulls the full sglang runtime dependency set (py-spy, pyzmq,
quack-kernels, pydantic, ...) and fails during environment preparation on
runners without a GPU.

## Configuration

`pytest.ini` in this directory pins `rootdir`/`testpaths` to this directory so
the repository-root pytest configuration is not picked up. Always pass
`-c tests/quantization/pytest.ini` when invoking pytest for this suite.
