# tests/quantization

Self-contained tests for explicit activation-quantization fallback handling
(`VLLM_STRICT_QUANT_SCHEME`).

## Why this directory is special

The activation-fallback suite (`test_activation_fallback.py`) is deliberately
self-contained: it loads the module under test directly from its file path and
only uses the standard library plus `pytest`. It must **not** require a full
`pip install -e python` of sglang (no GPU, no compiled kernels, no third-party
runtime deps).

## Running the suite

Install only the minimal dependencies and run pytest with this directory as the
rootdir so that `tests/quantization/pytest.ini` is picked up instead of the
repository-wide configuration:

```bash
pip install -r tests/quantization/requirements.txt
python -m pytest -q tests/quantization/test_activation_fallback.py
```

Stability check (no random ordering plugin):

```bash
python -m pytest -q -p no:randomly tests/quantization/test_activation_fallback.py
```

Performance check:

```bash
python -m pytest -q tests/quantization/test_activation_fallback.py --durations=10
```

Usability check (strict-mode env var):

```bash
python -m pytest -q tests/quantization/test_activation_fallback.py -k strict
```

## CI guidance

CI jobs that run this directory should install only
`tests/quantization/requirements.txt` (i.e. `pytest`) and must **not** run
`pip install -e python`, which pulls the full sglang runtime dependency set
(including CUDA/kernel packages) and fails on GPU-less runners.

The suite requires the source tree to be present (it reads
`python/sglang/srt/layers/quantization/activation_fallback.py` from disk) but
it does not import the `sglang` package.
