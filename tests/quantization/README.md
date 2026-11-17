# Self-contained quantization tests

This directory hosts the activation-quantization fallback suite
(`test_activation_fallback.py`). It is deliberately self-contained so it can
run on GPU-less CI runners without installing the full `sglang` runtime
dependency set.

## Why self-contained

The module under test
(`python/sglang/srt/layers/quantization/activation_fallback.py`) only uses the
standard library. The test module loads it directly from its file path, so no
`import sglang` is required. This avoids the failure mode where CI runs
`pip install -e python`, which resolves the full sglang dependency set and
fails on runners without a GPU or the required system libraries.

## How to run

Install only the minimal dependencies:

```bash
pip install -r tests/quantization/requirements.txt
```

Then run the suite with the directory-local pytest configuration so the
repository-root `pytest.ini` is not picked up:

```bash
python -m pytest -q -c tests/quantization/pytest.ini tests/quantization
```

Do **not** run `pip install -e python` (or `pip install sglang`) before this
suite. That pulls the full runtime dependency set and is exactly the
environment-setup failure this suite is designed to avoid.

## CI guidance

CI jobs that run this directory should:

1. Install only `tests/quantization/requirements.txt` (pytest and
   pytest-randomly).
2. Set `PYTHONPATH` to the repository `python/` directory if any future test
   in this directory needs to import `sglang` from the checkout.
3. Invoke pytest with `-c tests/quantization/pytest.ini` so rootdir/testpaths
   are pinned to this directory.

If the CI configuration cannot be changed, provide reproducible local
verification evidence (full commands and output) in the commit description.
