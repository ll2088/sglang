# tests/quantization

Self-contained test suite for explicit activation-quantization fallback
handling (`VLLM_STRICT_QUANT_SCHEME`).

## Why this directory is self-contained

The suite under test (`python/sglang/srt/layers/quantization/activation_fallback.py`)
only uses the standard library. The tests load it directly from its file path
so they can run without a GPU, without compiled kernels, and without a full
`pip install -e python` of sglang.

## How to run

From the repository root:

```bash
python -m pytest -q -c tests/quantization/pytest.ini tests/quantization/test_activation_fallback.py
```

Install only the minimal dependencies:

```bash
pip install -r tests/quantization/requirements.txt
```

Do **not** run `pip install -e python` for this suite. That pulls the full
sglang runtime dependency set (py-spy, pybase64, pydantic, pyzmq,
quack-kernels, regex, ...) and fails on GPU-less runners.

## CI notes

- Pin the Python version used by the job (the suite targets CPython 3.10+).
- Set `PYTHONPATH` to the repository root (and optionally `python/`) so the
  source tree is importable if a future test needs it.
- The `conftest.py` in this directory fails fast with an actionable message if
  the module under test is missing from the checkout.
