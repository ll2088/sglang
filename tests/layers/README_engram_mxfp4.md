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

Point `PYTHONPATH` at the source directory and run the self-check directly.
Use `--confcutdir=tests/layers` so the repository root `conftest.py` (which
may import `sglang`) is not loaded, and `-p no:cacheprovider` to avoid
writing cache files in a read-only CI checkout:

```sh
PYTHONPATH=python python -m pytest -q --confcutdir=tests/layers \
    -p no:cacheprovider tests/layers/test_engram_mxfp4_selfcheck.py
```

Or run the stdlib-only self-check without pytest at all:

```sh
PYTHONPATH=python python tests/layers/test_engram_mxfp4_selfcheck.py
```

Both commands load `engram_mxfp4.py` by file path and never import the
`sglang` package, so they work even when `pip install -e python` fails during
dependency resolution. CI must use these commands instead of any step that
runs `pip install -e python` or otherwise resolves the full sglang dependency
tree (py-spy, pybase64, pydantic, quack-kernels, ...).

## Minimal CI environment

CI environment preparation must not install the `sglang` package itself for
these tests. Install only the minimal test dependency set and run the tests
with `PYTHONPATH=python`. The `tests/layers/pytest.ini` file is intentionally
minimal and stdlib-only; pass it explicitly with `-c tests/layers/pytest.ini`
(or use `--confcutdir=tests/layers`) so the repository root configuration is
not loaded:

```sh
python -m pip install --no-deps pytest
PYTHONPATH=python python -m pytest -q -c tests/layers/pytest.ini \
    --confcutdir=tests/layers -p no:cacheprovider \
    tests/layers/test_engram_mxfp4_minimal.py
```

The `test_engram_mxfp4_minimal.py` entry point is self-contained: it loads the
layout module and the stdlib-only self-check by file path and never imports
`sglang`, so it stays runnable even when the repository root `conftest.py`
cannot be imported.

The `--no-deps` flag prevents pip from resolving the full sglang dependency
tree (py-spy, pybase64, pydantic, quack-kernels, ...). The `tests/layers`
`conftest.py` registers the stdlib-only marker and keeps the source directory
importable without importing `sglang`, so collection succeeds in a minimal
environment. If the full dependency tree is available, the torch-dependent
cases in `test_engram_mxfp4.py` run as well.

## Running the full suite (torch available)

When the full dependency tree is installed, the torch-dependent cases run too:

```sh
python -m pytest -q tests/layers/test_engram_mxfp4_selfcheck.py tests/layers/test_engram_mxfp4.py
```

The torch-dependent cases are skipped individually when `torch` is missing, so
the stdlib-only subset always runs.
