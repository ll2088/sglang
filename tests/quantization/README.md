# Activation-quantization fallback suite

Self-contained tests for `python/sglang/srt/layers/quantization/activation_fallback.py`.

## Why this suite is isolated

The module under test only uses the standard library. Importing `sglang` (or any
of its submodules) resolves the full runtime dependency set (`py-spy`,
`pybase64`, `pydantic`, `pyzmq`, `quack-kernels`, `regex`, ...), which is what
made CI environment preparation fail. This suite therefore runs with **pytest
alone**.

## Running locally / in CI

```bash
pip install -r tests/quantization/requirements.txt
python -m pytest -q -c tests/quantization/pytest.ini \
    tests/quantization/test_activation_fallback.py
```

Stability check (no random ordering):

```bash
python -m pytest -q -c tests/quantization/pytest.ini -p no:randomly \
    tests/quantization/test_activation_fallback.py
```

Performance check:

```bash
python -m pytest -q -c tests/quantization/pytest.ini \
    tests/quantization/test_activation_fallback.py --durations=10
```

Usability check (strict mode):

```bash
python -m pytest -q -c tests/quantization/pytest.ini \
    tests/quantization/test_activation_fallback.py -k strict
```

## CI requirements

* Install **only** `tests/quantization/requirements.txt` (pytest).
* Do **not** run `pip install -e python` / `pip install -e .` for this suite.
* Always pass `-c tests/quantization/pytest.ini` so the repository-root
  `pyproject.toml` / `setup.py` is not consulted.
* `conftest.py` strips the repository root and any `sglang` path from
  `sys.path` and `PYTHONPATH` before collection, and loads the module under
  test directly from its source file.
