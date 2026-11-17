# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) vLLM project
#
# Minimal-dependency entry point for the activation-fallback suite.
#
# CI must use this target (or an equivalent command) instead of
# ``pip install -e python`` for tests/quantization. Installing the sglang
# package resolves its full dependency set (py-spy, pybase64, pydantic,
# pyzmq, quack-kernels, ...) and fails during environment preparation.
#
# The module under test only uses the standard library, so pytest is the only
# dependency required.

.PHONY: test-quantization test-quantization-strict test-quantization-stability

# Install only the minimal dependencies and run the activation-fallback suite.
# ``--no-deps`` prevents pip from resolving the sglang package metadata.
test-quantization:
	python -m pip install --no-deps -r tests/quantization/requirements.txt
	python -m pytest -q -c tests/quantization/pytest.ini tests/quantization/test_activation_fallback.py

# Run only the strict-mode tests (VLLM_STRICT_QUANT_SCHEME).
test-quantization-strict:
	python -m pip install --no-deps -r tests/quantization/requirements.txt
	python -m pytest -q -c tests/quantization/pytest.ini tests/quantization/test_activation_fallback.py -k strict

# Run the suite three times with random ordering disabled to catch order
# dependencies or flaky failures.
test-quantization-stability:
	python -m pip install --no-deps -r tests/quantization/requirements.txt
	python -m pytest -q -c tests/quantization/pytest.ini -p no:randomly tests/quantization/test_activation_fallback.py
	python -m pytest -q -c tests/quantization/pytest.ini -p no:randomly tests/quantization/test_activation_fallback.py
	python -m pytest -q -c tests/quantization/pytest.ini -p no:randomly tests/quantization/test_activation_fallback.py
