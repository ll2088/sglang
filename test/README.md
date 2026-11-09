# Test and Continuous Integration (CI) System in SGLang

This page covers principles and essentials: folder layout, how to run tests, registration, and suite selection. For complete references, see the skill guides:

- **Writing tests** — templates, fixtures, model selection, complete suite tables, checklist: [`.claude/skills/write-sglang-test/SKILL.md`](../.claude/skills/write-sglang-test/SKILL.md)
- **CI pipeline internals** — stage flow diagrams, fail-fast layers, gating, partitioning, execution modes, debugging failures: [`.claude/skills/ci-workflow-guide/SKILL.md`](../.claude/skills/ci-workflow-guide/SKILL.md)

## CI Pipeline Overview

The CI pipeline runs in three sequential stages: **A** (pre-flight, ~3 min) → **B** (basic, ~30 min) → **C** (advanced, ~30 min). Kernel and multimodal-gen tests run in parallel with stage B. For details on stage gating, fail-fast mechanisms, execution modes (PR vs scheduled vs manual dispatch), and debugging CI failures, see the [CI workflow guide](../.claude/skills/ci-workflow-guide/SKILL.md).

## Folder Organization

- `registered/`: CI test files, including kernel tests and benchmarks, auto-discovered by `run_suite.py`.
- `manual/`: Non-CI tests for local debugging or special setups.
- `run_suite.py`: CI runner — scans `registered/` recursively.

The system supports both [unittest](https://docs.python.org/3/library/unittest.html) and [pytest](https://docs.pytest.org/en/stable/). The launcher runs `python filename.py -f` with **failfast enabled by default**.

Make sure your file ends with **exactly** one of:

```python
# for unittest
if __name__ == "__main__":
    unittest.main()
```

```python
# for pytest
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
```

Do not add custom `argparse` or modify `sys.argv` before these calls — the CI runner appends `-f` for failfast.

## Run Tests Locally

```bash
# Single file
python3 test/registered/core/test_srt_endpoint.py

# Single test method
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Single JIT kernel test
python3 test/registered/kernels/ops/elementwise/test_add_constant.py

# Run a suite
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu
python3 test/run_suite.py --hw cuda --suite base-a-test-1-gpu-small

# Nightly tests (CUDA nightly suites take no --nightly; the stage is in the name)
python3 test/run_suite.py --hw cuda --suite nightly-test-1-gpu-large

# With auto-partitioning (for parallel CI jobs)
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small \
    --auto-partition-id 0 --auto-partition-size 4
```

## Exit Codes

`run_suite.py` returns distinct exit codes so CI can tell collection problems
apart from real test failures:

| Code | Meaning |
|------|---------|
| `0` | All selected tests passed |
| `1` | Test execution failure |
| `2` | Collection failure (bad registration, import error, invalid suite) |
| `3` | No tests matched the requested suite/partition |

On a collection failure the runner prints a `COLLECTION FAILED` block with the
exception type, reason and traceback instead of silently passing. The three
non-zero states are also greppable as `collect-failed`, `exec-failed` and
`no-tests` so CI logs and docs can be checked mechanically.

In short: `Collection failure` (exit `2`) means the suite could not be
collected, `Test execution failure` (exit `1`) means collected tests failed,
and `No tests matched` (exit `3`) means the selection was empty.

To verify the runner without installing the full sglang dependency set:

```bash
bash test/unit/setup_env.sh
python test/unit/verify_run_suite.py
```

`verify_run_suite.py` accepts optional dimension names
(`collect-diagnostics`, `exit-codes`, `tests`, `usability`) and runs all four
when none are given. CI must call `test/unit/setup_env.sh` for environment
preparation instead of `pip install -e python`; the latter resolves the full
sglang dependency set and can fail before any test runs.

`test/unit/setup_env.sh` installs only `test/unit/requirements.txt` (pytest)
and self-checks `import pytest` / `pytest --version`, so an environment
problem is reported as an environment failure instead of being mistaken for a
test failure. Do not run `pip install -e python` for these checks: it resolves
the full sglang dependency set (torch, kernels, ...) and can fail before any
test runs.

`verify_run_suite.py` runs the collect-diagnostics, exit-codes, tests and
usability dimensions and exits `2` when the environment itself is broken
(infra failure) versus `1` for a real dimension failure.

The repository-root `setup_env.sh` delegates to `test/unit/setup_env.sh`, so
whichever setup entry CI calls, only `test/unit/requirements.txt` (pytest) is
installed. Both scripts print `[setup] using <resolved path>` so CI logs prove
the lightweight entry was hit instead of a full `pip install -e python`.

`test/unit/setup_env.sh` is the only supported environment-preparation entry
for these checks. It installs `test/unit/requirements.txt` (pytest only),
prints the resolved script path so CI logs prove it was hit, and self-checks
`import pytest` / `pytest --version`. CI must call this script instead of
`pip install -e python`; the latter resolves the full sglang dependency set
(torch, kernels, ...) and can fail before any test runs.

`test/unit/setup_env.sh` installs only `test/unit/requirements.txt` (pytest)
and self-checks `import pytest` / `pytest --version`, so an environment
problem is reported as an environment failure instead of being mistaken for a
test failure. Do not run `pip install -e python` for these checks: it resolves
the full sglang dependency set (torch, kernels, ...) and can fail before any
test runs.

`verify_run_suite.py` is the CI entrypoint for these checks. It runs the
collect-diagnostics, exit-codes, tests and usability dimensions, prints the
pytest version, and exits `2` when the environment itself is broken (infra
failure) versus `1` for a real dimension failure. CI must install
`test/unit/requirements.txt` (pytest only) instead of `pip install -e python`,
which resolves the full sglang dependency set and can fail before pytest
starts.

`verify_run_suite.py` runs the collect-diagnostics, exit-codes, tests and
usability dimensions, prints the pytest version and installed packages, and
exits `2` when the environment itself is broken (infra failure) versus `1` for
a real dimension failure.

## CI Registration

Every CI-discovered test file must call a registration function at module level:

```python
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-small")
```

Parameters: `est_time` (seconds), `stage` + `runner_config` (target stage and runner pool from `scripts/ci/runner_configs.yml`), `nightly=True` (nightly-only), `disabled="reason"` (temporarily disable).

Keep `est_time`, `stage`, `runner_config` as **literal values** — `run_suite.py` collects them by AST parsing.

Directories under `test/registered/` group tests by topic and are free-form
(`lora/`, `hicache/`, `disaggregation/`, `perf/`, ...); unit tests cover one srt
module, so they mirror the source tree under `unit/`. What a test costs, which
stage gates it and which runner it needs are declared by its `register_*_ci`
call -- including hardware, which is expressed by one or more `register_*_ci`
calls and never by a new top-level directory. Kernel tests use
`test/registered/kernels/{ops,benchmark}/<group>/`, retaining the established
plural `kernels` root.

Diffusion workflows also enter through `test/run_suite.py`; registered bridge
files preserve their case-level pytest partitioning until the remaining
diffusion cases are moved out of the package test-support tree.

Kernel correctness tests and benchmarks use the established plural `kernels`
root and mirror the operator group under `python/sglang/kernels/ops/`. Helpers
stay alongside the kernel source under `python/sglang/kernels/jit/` and are
imported by absolute path:

- Correctness tests: `test/registered/kernels/ops/<group>/test_*.py` → `base-b-kernel-unit-test-1-gpu-large`
- Benchmarks: `test/registered/kernels/benchmark/<group>/bench_*.py` → `base-b-kernel-benchmark-test-1-gpu-large`

## Choosing a Suite

Use the lightest suite that meets your test's needs. Full suite tables are in the [write-sglang-test skill](../.claude/skills/write-sglang-test/SKILL.md#all-ci-suites).

| Need | Suite |
|------|-------|
| No GPU required | `base-a-test-cpu` |
| Small GPU (fits 5090, 32GB) | `base-b-test-1-gpu-small` (most tests go here) |
| Large GPU memory or Hopper features | `base-b-test-1-gpu-large` |
| JIT kernel correctness | `base-b-kernel-unit-test-1-gpu-large` |
| JIT kernel benchmarks | `base-b-kernel-benchmark-test-1-gpu-large` |
| Multi-GPU (2/4/8) | `base-b-test-2-gpu-large`, `base-c-test-*` |
| Long-running or experimental | `nightly-*` suites |

## Steps for Adding a Test

See the [write-sglang-test skill](../.claude/skills/write-sglang-test/SKILL.md) for templates, fixtures, model selection, and a complete checklist.

Before adding a registered test, identify the production change that would make
it fail. Prefer extending an existing fixture/server launch over adding another
file. The incremental admission check applies these ratchets to new or modified
registered tests:

- Temporary `disabled=` registrations and unconditional skips must reference an
  issue and include `until YYYY-MM-DD`; expired entries fail lint.
- A file registered on CUDA plus another accelerator must place a nearby
  `backend-specific:` comment above the extra registration and name the path or
  failure mode that only that backend can catch.
- Default PR registrations are limited to 1,200 estimated weighted accelerator-seconds
  per backend (`est_time * GPU count`). Move larger matrices to extra/nightly,
  or document a nearby `ci-cost-override:` rationale.

## Multi-Hardware Backends

This README mostly describes the NVIDIA GPU CI pipeline. Other hardware backends (AMD, NPU) follow the same practices and use the multi-backend registry system. A scheduled job summarizes test coverage across all backends; [here is an example run](https://github.com/sgl-project/sglang/actions/runs/23424304300).

## Tips

- Learn from existing examples in [test/registered](https://github.com/sgl-project/sglang/tree/main/test/registered).
- Reuse servers — launching is expensive. Share one server across many test methods via `setUpClass`.
- Use as few GPUs as possible. Prefer 1-GPU runners.
- Each test file should take < 500 seconds; split if longer.
- Each GitHub Actions job should take < 30 minutes; split if longer.
- If tests are too slow for per-commit, consider nightly suites.
- New GSM8K or MMLU accuracy checks should use sgl-eval (`sglang.test.sgl_eval_utils.run_sgl_eval` or `sgl-eval run`) instead of adding another dataset or evaluator copy.

## Other Notes

### Adding New Models to Nightly CI
- **Text models**: Extend the [global model list variables](https://github.com/sgl-project/sglang/blob/85c1f7937781199203b38bb46325a2840f353a04/python/sglang/test/test_utils.py#L104) in `test_utils.py`.
- **VLMs**: Extend the `MODEL_THRESHOLDS` dictionary in `test/registered/accuracy/models/test_vlms_mmmu_eval.py`.
