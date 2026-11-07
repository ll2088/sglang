# Kernel tests

`test_dsa_topk_bf16.py` exercises the BF16 indexer top-k capability negotiation
and the host-side mirror of the BF16x2 packed histogram extraction.  It is
deliberately self-contained: it imports only `numpy` and `pytest`, so it can run
in a minimal CI job without installing the full `sglang` dependency tree.

Run it with:

```sh
python -m pytest -q tests/kernels/test_dsa_topk_bf16.py
```

Tests that need the compiled extension or a GPU should be marked with
`@pytest.mark.requires_sglang` and live in a separate job that installs the
runtime.
