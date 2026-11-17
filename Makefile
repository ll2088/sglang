# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) vLLM project
#
# Minimal entry points for the activation-fallback suite.
#
# CI must use these targets (or the equivalent commands) instead of
# ``pip install -e python``: the suite is self-contained and only needs
# pytest, so installing the full sglang dependency set is both unnecessary and
# the cause of the environment-preparation failure.

PYTHON ?= python
QUANT_TEST_DIR := tests/quantization
QUANT_TEST_INI := $(QUANT_TEST_DIR)/pytest.ini
QUANT_TEST_REQS := $(QUANT_TEST_DIR)/requirements.txt
QUANT_TEST_FILE := $(QUANT_TEST_DIR)/test_activation_fallback.py

.PHONY: install-quant-test-deps test-quant test-quant-strict test-quant-stability test-quant-perf

# Install ONLY the minimal requirements for the activation-fallback suite.
# Never run ``pip install -e python`` for this suite.
install-quant-test-deps:
	$(PYTHON) -m pip install -r $(QUANT_TEST_REQS)

# Run the activation-fallback suite with the dedicated pytest.ini so the
# repository-wide configuration (and its import paths) is not picked up.
test-quant:
	$(PYTHON) -m pytest -q -c $(QUANT_TEST_INI) $(QUANT_TEST_FILE)

# Usability check: VLLM_STRICT_QUANT_SCHEME behavior.
test-quant-strict:
	$(PYTHON) -m pytest -q -c $(QUANT_TEST_INI) $(QUANT_TEST_FILE) -k strict

# Stability check: run without the random-order plugin to confirm no ordering
# dependency, three times.
test-quant-stability:
	$(PYTHON) -m pytest -q -c $(QUANT_TEST_INI) -p no:randomly $(QUANT_TEST_FILE)
	$(PYTHON) -m pytest -q -c $(QUANT_TEST_INI) -p no:randomly $(QUANT_TEST_FILE)
	$(PYTHON) -m pytest -q -c $(QUANT_TEST_INI) -p no:randomly $(QUANT_TEST_FILE)

# Performance check: report the slowest tests to confirm no regression.
test-quant-perf:
	$(PYTHON) -m pytest -q -c $(QUANT_TEST_INI) $(QUANT_TEST_FILE) --durations=10
