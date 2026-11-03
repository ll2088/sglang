# Quantization regression tests

These tests are self-contained: they only need `pytest` and the Python
standard library. They must not import `sglang` or `torch`, so they can run in
a minimal CI environment without installing the full sglang dependency set.

## Running locally

```bash
bash scripts/ci/install_test_deps.sh
PYTHONPATH=python python -m pytest -q test/registered/quant
```

## Static review

```bash
bash scripts/ci/check_quant_patterns.sh
```

## CI notes

- Do **not** run `pip install -e python` for these tests; it pulls torch,
triton and the nvidia-* wheels, which is unnecessary and can fail on newer
Python versions.
- Use `PYTHONPATH=python` so the tests resolve modules from the source tree
without an editable install.
- The dependency files live only under `scripts/ci/`; do not duplicate them
under `test/registered/quant/`.
