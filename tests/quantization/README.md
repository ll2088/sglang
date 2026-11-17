# tests/quantization

Self-contained test suite for explicit activation-quantization fallback
handling (`VLLM_STRICT_QUANT_SCHEME`).

## Why this directory is self-contained

The suite loads the module under test directly from its file path
(`python/sglang/srt/layers/quantization/activation_fallback.py`) and only
uses the standard library plus pytest. It must run on GPU-less runners
without compiled kernels and without the full sglang runtime dependency
set.

## How to run

```bash
# Install only the minimal dependencies for this directory.
pip install -r tests/quantization/requirements.txt

# Run with the directory-local pytest configuration so that rootdir and
# testpaths are pinned to this directory.
python -m pytest -q -c tests/quantization/pytest.ini tests/quantization/test_activation_fallback.py
```

## CI guidance

CI jobs that run this directory must:

1. Install **only** `tests/quantization/requirements.txt` (pytest).
2. **Not** run `pip install -e python`; that pulls the full sglang runtime
   dependency set and fails on GPU-less runners before pytest can start.
3. Pass `-c tests/quantization/pytest.ini` so the directory-local
   configuration is used instead of the repository-root configuration.
4. Optionally set `PYTHONPATH` to the repository `python/` source tree if a
   future test in this directory needs to import `sglang` from the checkout.

If the environment preparation step fails, print the full pip error output
(including the failing requirement and the Python version) so the real
cause (dependency resolution conflict, Python version mismatch, or index
/ network issue) is visible in CI logs.
