# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) vLLM project
#
# Minimal, self-contained targets for the activation-fallback suite.
#
# CI must call ``make test-quantization`` (or the equivalent commands below)
# instead of running ``pip install -e python`` for this suite: the latter
# resolves the full sglang dependency set (py-spy, pybase64, pydantic,
# quack-kernels, ...) during environment preparation and fails before any test
# executes. The suite only needs pytest and is loaded from source file paths.

PYTHON ?= python
QUANT_DIR := tests/quantization
QUANT_REQ := $(QUANT_DIR)/requirements.txt
QUANT_INI := $(QUANT_DIR)/pytest.ini
QUANT_TEST := $(QUANT_DIR)/test_activation_fallback.py

.PHONY: test-quantization test-quantization-stability test-quantization-perf

# Install only the minimal dependency set, then run the suite with an explicit
# config so pytest never walks up to the repository-root pyproject.toml.
test-quantization:
	$(PYTHON) -m pip install -r $(QUANT_REQ)
	PYTHONPATH= $(PYTHON) -m pytest -q -c $(QUANT_INI) $(QUANT_TEST)

# Repeat the suite without random ordering to catch order dependencies.
test-quantization-stability:
	$(PYTHON) -m pip install -r $(QUANT_REQ)
	PYTHONPATH= $(PYTHON) -m pytest -q -c $(QUANT_INI) -p no:randomly $(QUANT_TEST)

# Report the slowest tests so reviewers can compare against a baseline.
test-quantization-perf:
	$(PYTHON) -m pip install -r $(QUANT_REQ)
	PYTHONPATH= $(PYTHON) -m pytest -q -c $(QUANT_INI) $(QUANT_TEST) --durations=10
