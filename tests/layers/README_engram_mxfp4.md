# MXFP4 Engram layout tests (dependency-free entry point)

These tests cover the MXFP4-packed Engram table layout parsing, E2M1 decode,
row-width aware allocation, and lookup consistency between packed and unpacked
tables. They mirror the upstream vLLM #61530 capability.

## Why a dependency-free entry point exists

CI environment preparation can fail while installing the full `sglang`
dependency tree (`torch`, `triton`, `py-spy`, `pybase64`, `pydantic`,
`quack-kernels`, ...). The layout module `python/sglang/srt/layers/engram_mxfp4.py`
is stdlib-only, so the tests for it must not require that install.

## Running without installing sglang

Point `PYTHONPATH` at the source directory and run the self-check directly:

```sh
PYTHONPATH=python python -m pytest -q tests/layers/test_engram_mxfp4_selfcheck.py
```

Or run the stdlib-only self-check without pytest at all:

```sh
PYTHONPATH=python python tests/layers/test_engram_mxfp4_selfcheck.py
```

Both commands load `engram_mxfp4.py` by file path and never import the
`sglang` package, so they work even when `pip install -e python` fails during
dependency resolution.

## Running the full suite (torch available)

When the full dependency tree is installed, the torch-dependent cases run too:

```sh
python -m pytest -q tests/layers/test_engram_mxfp4_selfcheck.py tests/layers/test_engram_mxfp4.py
```

The torch-dependent cases are skipped individually when `torch` is missing, so
the stdlib-only subset always runs.
